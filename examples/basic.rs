// trait Determinant {
//     fn determinant(&self) -> f32;
// }

// struct Mat3([[f32; 3]; 3]);
// struct Mat4([[f32; 4]; 4]);

// =============================================================================

use tetra_test::*;

fn main() {
    let points = vec![
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
        points.into_iter(),
    );

    println!("{delaunay:?}");
}
