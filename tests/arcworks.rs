#[test]
fn prime_field_17_aritmetric() {
    use ark_ff::fields::{Field, Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "17"]
    #[generator = "3"]
    pub struct FqConfig;
    pub type Fq = Fp64<MontBackend<FqConfig, 1>>;

    let a = Fq::from(9);
    let b = Fq::from(10);

    assert_eq!(a, Fq::from(26)); // 26 =  9 mod 17
    assert_eq!(a - b, Fq::from(16)); // -1 = 16 mod 17
    assert_eq!(a + b, Fq::from(2)); // 19 =  2 mod 17
    assert_eq!(a * b, Fq::from(5)); // 90 =  5 mod 17
    assert_eq!(a.square(), Fq::from(13)); // 81 = 13 mod 17
    assert_eq!(b.double(), Fq::from(3)); // 20 =  3 mod 17
    assert_eq!(a / b, a * b.inverse().unwrap()); // need to unwrap since `b` could be 0 which is not invertible
}
