use three_delaunay::*;

fn main() {
    let points = vec![
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
        points.into_iter(),
    )
    .expect("compute tetra");

    println!("{:?}", delaunay.points());
    println!("{:?}", delaunay.tetra());
}
