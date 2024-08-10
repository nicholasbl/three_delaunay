use std::ops::{Add, Div, Sub};

use glam::{u64vec3, BVec3A, DVec3, IVec3, Mat3A, U64Vec3, Vec3, Vec3A, Vec3Swizzles};
use kiddo::SquaredEuclidean;

#[derive(Debug)]
pub struct Tetrahedron {
    vertices: [Vec3A; 4],
}

impl Tetrahedron {
    pub fn new(vertices: [Vec3A; 4]) -> Tetrahedron {
        Self { vertices }
    }

    // Check if a point is inside the circumsphere of the tetrahedron
    fn in_circumsphere(&self, point: Vec3A) -> bool {
        let (center, radius) = self.compute_circumsphere();
        return (point - center).length() < radius;
    }

    // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
    pub fn compute_circumsphere(&self) -> (Vec3A, f32) {
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

    fn volume(&self) -> f32 {
        let a = self.vertices[1] - self.vertices[0];
        let b = self.vertices[2] - self.vertices[0];
        let c = self.vertices[3] - self.vertices[0];

        (a.cross(b)).dot(c) / 6.0
    }

    fn to_tetra_coords(&self) -> Mat3A {
        let a = self.vertices[1] - self.vertices[0];
        let b = self.vertices[2] - self.vertices[0];
        let c = self.vertices[3] - self.vertices[0];

        glam::mat3a(a, b, c).transpose().inverse()
    }

    fn test_point(&self, point: Vec3A) -> bool {
        let new_p = self.to_tetra_coords() * point;
        new_p.cmple(Vec3A::ONE).all()
            && new_p.cmpge(Vec3A::ZERO).all()
            && new_p.element_sum() <= 1.0
    }
}

// =============================================================================
#[derive(Debug, Clone)]
pub struct TetrahedronIndex {
    vertices: [u32; 4],
}

impl TetrahedronIndex {
    pub fn new(vertices: [u32; 4]) -> Self {
        Self { vertices }
    }

    fn faces(&self) -> [Face; 4] {
        return [
            Face::new([self.vertices[0], self.vertices[1], self.vertices[2]]),
            Face::new([self.vertices[0], self.vertices[1], self.vertices[3]]),
            Face::new([self.vertices[0], self.vertices[2], self.vertices[3]]),
            Face::new([self.vertices[1], self.vertices[2], self.vertices[3]]),
        ];
    }
}

// =============================================================================

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

#[derive(Debug)]
struct DelaunayTriangulationProcess {
    points: Vec<Vec3A>,
    tetra: Vec<TetrahedronIndex>,
    faces: Vec<Face>,
    bad_tetrahedra: Vec<usize>,
}

impl DelaunayTriangulationProcess {
    fn new(points: Vec<Vec3A>, bounding_tetra: Vec<TetrahedronIndex>) -> Self {
        Self {
            points,
            tetra: bounding_tetra,
            faces: Default::default(),
            bad_tetrahedra: Default::default(),
        }
    }

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

    pub fn add_points<T: Iterator<Item = Vec3A>>(&mut self, from: T) {
        for p in from {
            self.add_point(p)
        }
    }

    fn add_point(&mut self, point: Vec3A) {
        self.bad_tetrahedra.clear();

        for (i, tetra) in self.tetra.iter().enumerate() {
            let realized = self.realize_tetra(tetra);

            if realized.in_circumsphere(point) {
                self.bad_tetrahedra.push(i);
            }
        }

        self.find_boundary_polygon();

        // Remove bad tetrahedra
        for &bad_tetra_index in self.bad_tetrahedra.iter().rev() {
            self.tetra.swap_remove(bad_tetra_index);
        }

        self.create_new_tetrahedra(point)
    }

    fn find_boundary_polygon(&mut self) {
        self.faces.clear();

        for &bad_tetra_index in &self.bad_tetrahedra {
            let bad_tetra = &self.tetra[bad_tetra_index];
            for i in 0..4 {
                for j in (i + 1)..4 {
                    for k in (j + 1)..4 {
                        let face = Face::new([
                            bad_tetra.vertices[i],
                            bad_tetra.vertices[j],
                            bad_tetra.vertices[k],
                        ]);

                        let is_shared = self.faces.iter().any(|f| f == &face);
                        if is_shared {
                            self.faces.retain(|f| f != &face);
                        } else {
                            self.faces.push(face);
                        }
                    }
                }
            }
        }
    }

    fn create_new_tetrahedra(&mut self, point: Vec3A) {
        let pid = self.points.len().try_into().unwrap();
        self.points.push(point);

        for face in &self.faces {
            let new_tetrahedron = TetrahedronIndex {
                vertices: [face.vtx[0], face.vtx[1], face.vtx[2], pid],
            };
            self.tetra.push(new_tetrahedron);
        }
    }
}

// =============================================================================

#[derive(Debug)]
pub struct DelaunayTetrahedra {
    points: Vec<Vec3A>,
    tetra: Vec<TetrahedronIndex>,
}

impl DelaunayTetrahedra {
    pub fn new<T: Iterator<Item = Vec3A>>(
        inital_points: Vec<Vec3A>,
        bounding_tetra: Vec<TetrahedronIndex>,
        from: T,
    ) -> Self {
        let mut p = DelaunayTriangulationProcess::new(inital_points, bounding_tetra);

        p.add_points(from);

        Self {
            points: p.points,
            tetra: p.tetra,
        }
    }

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
}

// =============================================================================

/// Take a 3d coordinate, and map to a linear address
#[inline]
fn d_index(at: U64Vec3, counts: U64Vec3) -> u32 {
    (at.x + (at.y * counts.y) + (at.z * counts.yz().element_product())) as u32
}

/// Map a position to a cell_id
#[inline]
fn map_position_to_cell_id(v: Vec3A, bb_min: Vec3A, cell_count: Vec3A) -> U64Vec3 {
    (v - bb_min / cell_count).floor().as_u64vec3()
}

struct AABB<T> {
    min: T,
    max: T,
    range: T,
}

impl<T> AABB<T>
where
    T: Sub<Output = T> + Copy + Add<Output = T> + Div<Output = T>,
    <T as Add>::Output: Div<T>,
    T: From<(f32, f32, f32)>,
    T: StrictBound,
{
    fn new(min: T, max: T) -> Self {
        Self {
            min,
            max,
            range: max - min,
        }
    }

    fn center(&self) -> <T as Div>::Output {
        let d: T = (2.0, 2.0, 2.0).into();
        self.min + self.range / d
    }

    fn contains(&self, value: T) -> bool {
        self.min.all_less_eq_than(&value) && !self.max.all_less_than(&value)
    }
}

trait StrictBound {
    fn all_less_than(&self, other: &Self) -> bool;
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

pub fn create_bounding_tetrahedra(
    bounding_box: &AABB<Vec3A>,
) -> ([Vec3A; 8], [TetrahedronIndex; 6]) {
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
        Vec3A::select(mask, bounding_box.max, bounding_box.min)
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

pub fn parallel_delaunay(bounding_box: AABB<Vec3A>, all_points: &[Vec3A]) {
    let (initial_points, initial_tetra) = create_bounding_tetrahedra(&bounding_box);

    let bounding_range = bounding_box.range.as_dvec3();

    let cell_counts = {
        const AVERAGE_POINT_IN_CELL: i32 = 8;

        let num_points = all_points.len() as f64;

        let denom = 3.0 * (AVERAGE_POINT_IN_CELL as f64) * bounding_range.element_product();

        let beta = num_points / denom;

        (bounding_range * beta).ceil().as_u64vec3()
    };

    // Our work granularity is one cell

    // we need to create zones

    let cell_size = cell_counts.as_dvec3() / bounding_range;
    let cell_count_as_float = cell_counts.as_vec3a();

    // assign points to a cell
    let point_id_to_cell_id: Vec<_> = all_points
        .iter()
        .map(|f| {
            d_index(
                map_position_to_cell_id(*f, bounding_box.min, cell_count_as_float),
                cell_counts,
            )
        })
        .collect();

    let piter = all_points
        .iter()
        .enumerate()
        .map(|(i, f)| ([f.x, f.y, f.z], i as u64));

    let tree = kiddo::KdTree::<f32, 3>::from_iter(piter);

    let context = ParaContext {
        bounding_box,
        cell_counts,
        cell_size,
        tree,
        point_id_to_cell_id: &point_id_to_cell_id,
        points: &all_points,
        initial_tetra: &initial_tetra,
        bounding_points: &initial_points,
    };

    for i in 0..cell_counts.x {
        for j in 0..cell_counts.y {
            for k in 0..cell_counts.z {
                let cell_coordinate = u64vec3(i, j, k);

                compute_cell(&context, cell_coordinate)
            }
        }
    }
}

struct ParaContext<'a> {
    bounding_box: AABB<Vec3A>,
    cell_counts: U64Vec3,
    cell_size: DVec3,
    tree: kiddo::KdTree<f32, 3>,
    point_id_to_cell_id: &'a [u32],
    points: &'a [Vec3A],
    bounding_points: &'a [Vec3A],
    initial_tetra: &'a [TetrahedronIndex],
}

impl<'a> ParaContext<'a> {
    fn cell_bounds(&self, cell_coordinate: U64Vec3) -> AABB<Vec3A> {
        let min = self.bounding_box.min.as_dvec3() + cell_coordinate.as_dvec3() * self.cell_size;
        let max = min + self.cell_size;

        let min = min.as_vec3a();
        let max = max.as_vec3a();

        AABB::new(min, max)
    }

    fn for_points_in(&self, bounds: AABB<Vec3A>, mut func: impl FnMut(u64) -> ()) {
        let sphere_center = bounds.center();
        let r: &[f32; 3] = sphere_center.as_ref();
        let sphere_range = bounds.range.max_element() / 2.0;
        for item in self
            .tree
            .within_unsorted_iter::<SquaredEuclidean>(r, sphere_range)
        {
            if bounds.contains(self.points[item.item as usize]) {
                func(item.item)
            }
        }
    }
}

fn compute_cell(context: &ParaContext, cell_coordinate: U64Vec3) {
    let cell_id = d_index(cell_coordinate, context.cell_counts);

    let bounds = context.cell_bounds(cell_coordinate);

    // get all points we are going to need here

    let mut our_points = Vec::new();
    let mut our_points_id = Vec::new();

    context.for_points_in(bounds, |point_id| {
        our_points.push(context.points[point_id as usize]);
        our_points_id.push(point_id);
        assert!(cell_id == context.point_id_to_cell_id[point_id as usize]);
    });

    let tetra = DelaunayTetrahedra::new(
        context.bounding_points.to_owned(),
        context.initial_tetra.to_owned(),
        our_points.into_iter(),
    );
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

    impl AlmostEqual for f32 {
        fn almost_equal(&self, other: Self, epsilon: Self) -> bool {
            (self - other).abs() < epsilon
        }
    }

    fn random_sphere_point() -> Vec3A {
        let x: f32 = rand::thread_rng().sample(StandardNormal);
        let y: f32 = rand::thread_rng().sample(StandardNormal);
        let z: f32 = rand::thread_rng().sample(StandardNormal);

        let mut v: Vec3A = (x, y, z).into();

        if v.length() == 0.0 {
            v += Vec3A::new(0.01, 0.01, 0.01);
        }

        v.normalize()
    }

    fn random_sphere_tet() -> (Tetrahedron, Vec3A, f32) {
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
        let t = Tetrahedron {
            vertices: [
                (0.0, 0.0, 0.0).into(),
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into(),
            ],
        };

        let inside = Vec3A::splat(0.1);
        let outside = Vec3A::splat(2.0);

        assert!(t.in_circumsphere(inside));
        assert!(!t.in_circumsphere(outside));
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

        let inside = Vec3A::splat(0.1);
        let outside = Vec3A::splat(2.0);

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
        let points: Vec<Vec3A> = vec![
            (0.0, 0.0, 0.0).into(),
            (0.0, 0.0, -0.5).into(),
            (0.0, 0.2, 0.2).into(),
        ];

        let delaunay = DelaunayTetrahedra::new(
            vec![
                (-1.0, -1.0, -1.0).into(),
                (1.0, -1.0, -1.0).into(),
                (0.0, 1.0, -1.0).into(),
                (0.0, 0.0, 1.0).into(),
            ],
            vec![TetrahedronIndex::new([0, 1, 2, 3])],
            points.iter().cloned(),
        );

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

        // Check that no point is inside the circumsphere of any tetrahedron
        for tetra in &delaunay.tetra {
            let realized = delaunay.realize_tetra(tetra);
            for point in &points {
                let cs = realized.compute_circumsphere();

                if realized.in_circumsphere(*point) {
                    panic!(
                        "Point in circumsphere: {}, Center: {}, Radius: {}, Test: {}, Tetra: {:?}",
                        point,
                        cs.0,
                        cs.1,
                        (*point - cs.0).length(),
                        realized
                    )
                }
            }
        }
    }
}
