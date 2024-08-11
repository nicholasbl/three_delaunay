//! Compute 3D Delaunay tetrahedra of arbitrary points. For usage, see the
//! documentation for [`compute_tetrahedra`].

use std::ops::{Add, Div, Sub};

use glam::{DMat3, DVec3, Vec3A};
use rstar::{
    primitives::{GeomWithData, Rectangle},
    RTree,
};

/// A Tetrehedron, represented as 4x 3D points.
#[derive(Debug)]
pub struct Tetrahedron {
    vertices: [DVec3; 4],
}

impl Tetrahedron {
    /// Creates a new [`Tetrahedron`] from a set of four points.
    pub fn new(vertices: [DVec3; 4]) -> Tetrahedron {
        Self { vertices }
    }

    /// Check if a point is inside the circumsphere of this [`Tetrahedron`].
    fn in_circumsphere(&self, point: DVec3) -> bool {
        let (center, radius) = self.compute_circumsphere();
        (point - center).length() < radius
    }

    /// Returns the circumsphere of this [`Tetrahedron`], as a point and radius.
    pub fn compute_circumsphere(&self) -> (DVec3, f64) {
        // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
        let u1 = self.vertices[1] - self.vertices[0];
        let u2 = self.vertices[2] - self.vertices[0];
        let u3 = self.vertices[3] - self.vertices[0];

        let l01_sq = u1.length_squared();
        let l02_sq = u2.length_squared();
        let l03_sq = u3.length_squared();

        // center coords
        let circum_o = self.vertices[0]
            + (l01_sq * (u2.cross(u3)) + l02_sq * (u3.cross(u1)) + l03_sq * (u1.cross(u2)))
                / (2.0 * u1.dot(u2.cross(u3)));

        let p = circum_o - self.vertices[0];
        // radius
        let radius = p.length();

        (circum_o, radius)
    }

    /// Returns the volume of this [`Tetrahedron`].
    pub fn volume(&self) -> f64 {
        let a = self.vertices[1] - self.vertices[0];
        let b = self.vertices[2] - self.vertices[0];
        let c = self.vertices[3] - self.vertices[0];

        (a.cross(b)).dot(c) / 6.0
    }

    /// Compute the tetrahedra-local (barycentric) coordinate matrix for this [`Tetrahedron`].
    pub fn to_tetra_coords(&self) -> DMat3 {
        let a = self.vertices[1] - self.vertices[0];
        let b = self.vertices[2] - self.vertices[0];
        let c = self.vertices[3] - self.vertices[0];

        glam::dmat3(a, b, c).transpose().inverse()
    }

    /// Test if a point is inside this [`Tetrahedron`]. This can be expensive to call repeatedly.
    pub fn test_point(&self, point: DVec3) -> bool {
        let new_p = self.to_tetra_coords() * point;
        new_p.cmple(DVec3::ONE).all()
            && new_p.cmpge(DVec3::ZERO).all()
            && new_p.element_sum() <= 1.0
    }

    /// Returns the bounding box of the circumsphere of this [`Tetrahedron`].
    pub fn bounding_circum_box(&self) -> AABB<DVec3> {
        let (point, radius) = self.compute_circumsphere();

        AABB::new(point - radius, point + radius)
    }
}

// =============================================================================

/// A tetrahedron, represented as 4x indicies into a point array.
#[derive(Debug, Clone)]
pub struct TetrahedronIndex {
    vertices: [u32; 4],
}

impl TetrahedronIndex {
    pub fn new(vertices: [u32; 4]) -> Self {
        Self { vertices }
    }

    #[allow(unused)]
    fn faces(&self) -> [Face; 4] {
        [
            Face::new([self.vertices[0], self.vertices[1], self.vertices[2]]),
            Face::new([self.vertices[0], self.vertices[1], self.vertices[3]]),
            Face::new([self.vertices[0], self.vertices[2], self.vertices[3]]),
            Face::new([self.vertices[1], self.vertices[2], self.vertices[3]]),
        ]
    }
}

impl From<[u32; 4]> for TetrahedronIndex {
    fn from(value: [u32; 4]) -> Self {
        Self::new(value)
    }
}

// =============================================================================

/// A face of a tetrahedron, represented as a three-tuple of indicies into a point array
#[derive(Debug)]
struct Face {
    vtx: [u32; 3],
}

impl Face {
    fn new(mut idx: [u32; 3]) -> Self {
        idx.sort();
        Self { vtx: idx }
    }
}

impl PartialEq for Face {
    fn eq(&self, other: &Self) -> bool {
        self.vtx == other.vtx
    }
}

// =============================================================================

/// Typedef for a point in our tree
type TreePoint = [f64; 3];

/// Typedef for a cube in our tree
type TreeRect = Rectangle<TreePoint>;

/// Typedef for a node in the tree, holding a cube and an index
type IndexRect = GeomWithData<TreeRect, usize>;

// =============================================================================

/// State and caches for the tetrahedra process
#[derive(Debug)]
struct DelaunayProcess {
    /// Vertex locations
    points: Vec<DVec3>,
    /// List of active tetrahedra. These are options for 'deletion'
    tetra: Vec<Option<TetrahedronIndex>>,
    /// Cache for face computation
    faces: Vec<Face>,
    /// Cache for tetrahedra to remove, by index
    bad_tetrahedra: Vec<usize>,
    /// Cache for tetrahedra to remove, by tree value
    bad_tree: Vec<IndexRect>,
    /// Spatial tooling to speed up tetra lookups
    lookup_accel: RTree<IndexRect>,
}

impl DelaunayProcess {
    fn new(points: Vec<DVec3>, bounding_tetra: Vec<TetrahedronIndex>) -> Self {
        // Set up cache
        let mut ret = Self {
            points,
            tetra: bounding_tetra.into_iter().map(Some).collect(),
            faces: Default::default(),
            bad_tetrahedra: Default::default(),
            bad_tree: Default::default(),
            lookup_accel: Default::default(),
        };

        // Set up initial accel structure
        for (bounding_i, bounding) in ret.tetra.iter().enumerate() {
            let bb = ret
                .realize_tetra(bounding.as_ref().unwrap())
                .bounding_circum_box();

            ret.lookup_accel.insert(IndexRect::new(
                TreeRect::from_corners(bb.min.into(), bb.max.into()),
                bounding_i,
            ));
        }

        ret
    }

    /// Turn an index based tetra into an actual tetra with vertex info
    fn realize_tetra(&self, tetra: &TetrahedronIndex) -> Tetrahedron {
        Tetrahedron {
            vertices: [
                self.points[tetra.vertices[0] as usize],
                self.points[tetra.vertices[1] as usize],
                self.points[tetra.vertices[2] as usize],
                self.points[tetra.vertices[3] as usize],
            ],
        }
    }

    /// Add points from an iterator
    pub fn add_points<T: Iterator<Item = DVec3>>(&mut self, from: T) -> Result<(), TetraError> {
        for p in from {
            self.add_point(p)?
        }
        Ok(())
    }

    /// Add a single point
    fn add_point(&mut self, point: DVec3) -> Result<(), TetraError> {
        // Clear caches
        self.bad_tetrahedra.clear();
        self.bad_tree.clear();

        // Go through all tetra with a circumsphere that contains this point
        for rect in self.lookup_accel.locate_all_at_point(&point.into()) {
            let tetra = self.tetra[rect.data].as_ref().unwrap();
            let realized = self.realize_tetra(tetra);

            // Point is in this circumsphere, so we need to remove this tetra
            if realized.in_circumsphere(point) {
                self.bad_tetrahedra.push(rect.data);
                self.bad_tree.push(*rect);
            }
        }

        // Build face cache, or faces that we will need to build tetra out of
        self.find_boundary_polygon()?;

        // Remove bad tetrahedra
        for bad_tetra_index in &self.bad_tetrahedra {
            self.tetra[*bad_tetra_index] = None;
        }

        // We also need to erase the tetra in our accel tree
        for bad_tree_node in &self.bad_tree {
            self.lookup_accel.remove(bad_tree_node);
        }

        // Build the new tetra
        self.create_new_tetrahedra(point);

        Ok(())
    }

    /// Builds faces that will need to be connected to new tetra. This fills the faces cache, and uses the bad_tetrahedra cache
    ///
    fn find_boundary_polygon(&mut self) -> Result<(), TetraError> {
        self.faces.clear();

        for &bad_tetra_index in &self.bad_tetrahedra {
            let Some(bad_tetra) = &self.tetra[bad_tetra_index] else {
                // well, this would be bad
                return Err(TetraError::MissingTetra);
            };

            // Go over all vertex and create faces, and see who matches
            use itertools::Itertools;

            for (&i, &j, &k) in bad_tetra.vertices.iter().tuple_combinations() {
                let face = Face::new([i, j, k]);

                let is_shared = self.faces.iter().any(|f| f == &face);
                if is_shared {
                    // face is shared, remove all those that aren't equal
                    self.faces.retain(|f| f != &face);
                } else {
                    // keep face
                    self.faces.push(face);
                }
            }
        }

        Ok(())
    }

    /// Create a new tetrahedra from the face cache.
    fn create_new_tetrahedra(&mut self, point: DVec3) {
        let pid = self.points.len().try_into().unwrap();
        self.points.push(point);

        for face in &self.faces {
            let new_tetrahedron = TetrahedronIndex {
                vertices: [face.vtx[0], face.vtx[1], face.vtx[2], pid],
            };

            let bounds = self.realize_tetra(&new_tetrahedron).bounding_circum_box();

            let spot = self.tetra.len();

            self.tetra.push(Some(new_tetrahedron));

            self.lookup_accel.insert(IndexRect::new(
                TreeRect::from_corners(bounds.min.into(), bounds.max.into()),
                spot,
            ));
        }
    }
}

// =============================================================================

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TetraError {
    #[error("Internal error, missing tetrahedra!")]
    MissingTetra,
}

/// Computes tetrahedra of arbitrary points.
///
/// This function takes an initial set of tetrahedra (can be only one) which
/// must bound all subsequent points. This initial tetrahedra is/are then
/// modified incrementally with points as given from an iterator.
///
/// To create an initial set of tetrahedra from a bounding box, consider using
/// [`create_bounding_tetrahedra`].
///
/// # Errors
///
/// This function may return errors; some corner cases are not yet handled.
pub fn compute_tetrahedra<T>(
    inital_points: Vec<DVec3>,
    bounding_tetra: Vec<TetrahedronIndex>,
    other_points: T,
) -> Result<DelaunayTetrahedra, TetraError>
where
    T: Iterator<Item = DVec3>,
{
    let mut p = DelaunayProcess::new(inital_points, bounding_tetra);

    p.add_points(other_points)?;

    Ok(DelaunayTetrahedra {
        points: p.points,
        tetra: p.tetra.into_iter().flatten().collect(),
    })
}

/// Results of computing tetrahedra
#[derive(Debug)]
pub struct DelaunayTetrahedra {
    points: Vec<DVec3>,
    tetra: Vec<TetrahedronIndex>,
}

impl DelaunayTetrahedra {
    /// Returns a reference to the vertex locations of this [`DelaunayTetrahedra`] result.
    pub fn points(&self) -> &[DVec3] {
        &self.points
    }

    /// Returns a reference to the list of tetra of this [`DelaunayTetrahedra`] result. These tetra contain indices into the point list.
    pub fn tetra(&self) -> &[TetrahedronIndex] {
        &self.tetra
    }

    /// Convert an indexed tetra into one with vertex information
    pub fn realize_tetra(&self, tetra: &TetrahedronIndex) -> Tetrahedron {
        Tetrahedron {
            vertices: [
                self.points[tetra.vertices[0] as usize],
                self.points[tetra.vertices[1] as usize],
                self.points[tetra.vertices[2] as usize],
                self.points[tetra.vertices[3] as usize],
            ],
        }
    }
}

// =============================================================================

/// An axis-aligned bounding box
pub struct AABB<T> {
    min: T,
    max: T,
}

impl<T> AABB<T>
where
    T: Sub<Output = T> + Copy + Add<Output = T> + Div<Output = T>,
    <T as Add>::Output: Div<T>,
    T: From<(f64, f64, f64)>,
    T: StrictBound,
{
    fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

pub trait StrictBound {
    /// Check if all components of a vector are less than another
    fn all_less_than(&self, other: &Self) -> bool;

    /// Check if all components of a vector are greater than another
    fn all_less_eq_than(&self, other: &Self) -> bool;
}

impl StrictBound for Vec3A {
    fn all_less_than(&self, other: &Self) -> bool {
        self.cmplt(*other).all()
    }
    fn all_less_eq_than(&self, other: &Self) -> bool {
        self.cmple(*other).all()
    }
}

impl StrictBound for DVec3 {
    fn all_less_than(&self, other: &Self) -> bool {
        self.cmplt(*other).all()
    }
    fn all_less_eq_than(&self, other: &Self) -> bool {
        self.cmple(*other).all()
    }
}

/// Create an initial set of tetrahedra that cover a bounding box.
///
/// The result of this function can be directly used with [`compute_tetrahedra`]
pub fn create_bounding_tetrahedra(
    bounding_box: &AABB<DVec3>,
) -> ([DVec3; 8], [TetrahedronIndex; 6]) {
    let points = [
        [false, false, false], //0
        [false, false, true],  //1
        [false, true, false],  //2
        [false, true, true],   //3
        [true, false, false],  //4
        [true, false, true],   //5
        [true, true, false],   //6
        [true, true, true],    //7
    ]
    .map(|f| {
        let mask = f.into();
        DVec3::select(mask, bounding_box.max, bounding_box.min)
    });

    let tetra = [
        TetrahedronIndex::new([0, 1, 3, 7]),
        TetrahedronIndex::new([0, 2, 3, 7]),
        TetrahedronIndex::new([0, 1, 5, 7]),
        TetrahedronIndex::new([0, 4, 5, 7]),
        TetrahedronIndex::new([0, 2, 6, 7]),
        TetrahedronIndex::new([0, 4, 6, 7]),
    ];

    (points, tetra)
}

// =============================================================================

#[cfg(test)]
mod test {
    use rand::Rng;
    use rand_distr::StandardNormal;

    use super::*;

    trait AlmostEqual {
        fn almost_equal(&self, other: Self, epsilon: Self) -> bool;
    }

    impl AlmostEqual for f64 {
        fn almost_equal(&self, other: Self, epsilon: Self) -> bool {
            (self - other).abs() < epsilon
        }
    }

    fn random_sphere_point() -> DVec3 {
        let x: f64 = rand::thread_rng().sample(StandardNormal);
        let y: f64 = rand::thread_rng().sample(StandardNormal);
        let z: f64 = rand::thread_rng().sample(StandardNormal);

        let mut v: DVec3 = (x, y, z).into();

        if v.length() == 0.0 {
            v += DVec3::new(0.01, 0.01, 0.01);
        }

        v.normalize()
    }

    fn random_sphere_tet() -> (Tetrahedron, DVec3, f64) {
        let center_dist = rand::distributions::Uniform::from(-100.0..100.0);

        let center = (
            rand::thread_rng().sample(center_dist),
            rand::thread_rng().sample(center_dist),
            rand::thread_rng().sample(center_dist),
        )
            .into();

        let radius = rand::thread_rng().sample(rand::distributions::Uniform::from(1.0..100.0));

        let points = [0, 1, 2, 3].map(|_| random_sphere_point() * radius + center);
        (Tetrahedron::new(points), center, radius)
    }

    #[test]
    fn test_circum() {
        for _ in 0..20 {
            let (t, true_center, true_radius) = random_sphere_tet();

            let (computed_center, computed_radius) = t.compute_circumsphere();

            let delta = computed_center.distance(true_center);

            if delta > 0.003 {
                panic!(
                    "Distance err: {} vs {} = {}",
                    computed_center, true_center, delta
                )
            }

            if !computed_radius.almost_equal(true_radius, 0.002) {
                panic!(
                    "Radius err: {} vs {} = {}",
                    computed_radius,
                    true_radius,
                    (computed_radius - true_radius).abs()
                )
            }
        }
    }

    #[test]
    fn test_in_circumsphere() {
        let mut test_vec: Vec<(Tetrahedron, DVec3, f64)> = Vec::new();

        test_vec.push((
            Tetrahedron {
                vertices: [
                    (0.0, 0.0, 0.0).into(),
                    (1.0, 0.0, 0.0).into(),
                    (0.0, 1.0, 0.0).into(),
                    (0.0, 0.0, 1.0).into(),
                ],
            },
            glam::dvec3(0.5, 0.5, 0.5),
            0.866025,
        ));

        test_vec.push((
            Tetrahedron {
                vertices: [
                    (0.0, 0.1, 0.0).into(),
                    (-1.0, 0.0, 0.0).into(),
                    (0.0, 0.9, 0.0).into(),
                    (0.0, 0.0, 1.0).into(),
                ],
            },
            glam::dvec3(-0.545, 0.5, 0.545),
            0.868361,
        ));

        for (t, at, rad) in test_vec {
            let inside = at + DVec3::Z * (rad * 0.99);
            let outside = at + DVec3::Z * (rad * 1.01);
            assert!(t.in_circumsphere(inside));
            assert!(!t.in_circumsphere(outside));

            let cc = t.compute_circumsphere();

            assert!(cc.0.abs_diff_eq(at, 0.0001));
            assert!(cc.1.almost_equal(rad, 0.0001));
        }
    }

    #[test]
    fn test_in_tetra() {
        let t = Tetrahedron {
            vertices: [
                (0.0, 0.0, 0.0).into(),
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into(),
            ],
        };

        let inside = DVec3::splat(0.1);
        let outside = DVec3::splat(2.0);

        assert!(t.test_point(inside));
        assert!(!t.test_point(outside));
    }

    #[test]
    fn test_volume() {
        let t = Tetrahedron {
            vertices: [
                (0.0, 0.0, 0.0).into(),
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into(),
            ],
        };

        let volume = t.volume();
        let expected = 1.0 / 6.0;
        assert!(volume.almost_equal(expected, 1e-6));
    }

    #[test]
    fn test_delaunay_structure() {
        let points: Vec<DVec3> = vec![
            (0.0, 0.0, 0.0).into(),
            (0.0, 0.0, -0.5).into(),
            (0.0, 0.2, 0.2).into(),
        ];

        let delaunay = compute_tetrahedra(
            vec![
                (-1.0, -1.0, -1.0).into(),
                (1.0, -1.0, -1.0).into(),
                (0.0, 1.0, -1.0).into(),
                (0.0, 0.0, 1.0).into(),
            ],
            vec![TetrahedronIndex::new([0, 1, 2, 3])],
            points.iter().cloned(),
        )
        .expect("compute tetra");

        // Check no tetrahedra share more than one face
        for i in 0..delaunay.tetra.len() {
            for j in (i + 1)..delaunay.tetra.len() {
                let shared_faces = delaunay.tetra[i]
                    .faces()
                    .iter()
                    .filter(|&f| delaunay.tetra[j].faces().contains(f))
                    .count();
                assert!(shared_faces <= 1);
            }
        }

        // Check that all tetrahedra are defined by four unique points
        for tetra in &delaunay.tetra {
            let mut unique_vertices = std::collections::HashSet::new();
            for &vertex in &tetra.vertices {
                unique_vertices.insert(vertex);
            }
            assert_eq!(unique_vertices.len(), 4);
        }

        // Check that no point is well inside the circumsphere of any tetrahedron
        for (tetra_i, tetra) in delaunay.tetra.iter().enumerate() {
            let realized = delaunay.realize_tetra(tetra);
            for point in &points {
                let cs = realized.compute_circumsphere();

                let test = (*point - cs.0).length();

                let depth = test - cs.1 + 0.000001;

                if depth < 0.0 {
                    dbg!(&delaunay);
                    panic!(
                        "Point in circumsphere: {}, Center: {}, Radius: {}, Test: {}, Tetra: {:?}, ID: {tetra_i}, DEPTH {depth}",
                        point, cs.0, cs.1, test, realized
                    )
                }
            }
        }
    }
}
