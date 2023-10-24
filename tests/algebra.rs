use ark_ff::Field;
use ark_poly::univariate::DenseOrSparsePolynomial;
use ark_poly::Polynomial;
use ark_std::Zero;
use std::ops::{Add, Neg};

#[test]
fn exercise_33() {
    // Z5* = {1, 2, 3, 4}, Show that (Z5*, *) is a commutative group

    // 1,2,3,4 are all coprime to 5. Hence their multiplication is closed in a group Z5*
    // Due to multiplication this group is is commutative and associative
    // Neutral element is 1, because every element multiplied by 1 is itself
    // All elements are coprime to 5, so their inverse exists and is also in Z5*
    // Due to all this properties (Z5*, *) is a commutative group
}

#[test]
fn exercise_34() {
    // If n is a prime number, then all elements of Zn* are coprime to n.
    // Group neutral element is always 1
    // Inverse of every element exists and is also in Zn*. It can be computed
    // by the Fermat's little theorem as a^(n-2) mod n
}

#[test]
fn exercise_35() {
    // In a group (Zn, +) there are n possible remainders of division by n. Hence, order of group is n
}

#[test]
fn exercise_36() {
    // (Z6, +) = {0, 1, 2, 3, 4, 5}
    // 5, 5+5 = 4, 5+5+5=3, 5+5+5+5=2, 5+5+5+5+5=1, 5+5+5+5+5+5=0, 5 is generator of (Z6, +)
    // 2, 2+2=4, 2+2+2=0, 2+2+2+2=2, 2+2+2+2+2=4, 2+2+2+2+2+2=0, 2 is not generator of (Z6, +)
}

#[test]
fn exercise_38() {
    // Implement double and add algorithm
    // y = g*x (mod n)
    fn double_and_add(g: i64, mut x: i64, n: i64) -> i64 {
        let mut h = g;
        let mut y = 0;
        while x > 0 {
            if x & 1 == 1 {
                y = (y + h) % n;
            }
            h = (h + h) % n;
            x >>= 1;
        }
        y
    }

    assert_eq!(double_and_add(3, 4, 5), 2);
    assert_eq!(double_and_add(3, 6, 5), 3);
}

#[test]
fn exercise_39() {
    // Z5*[2] = {1, 4}
    // Associativity 1 * (1 * 4) == (1 * 1) * 4 == 4; 4 * (1 * 4) == (4 * 1) * 4 == 1
    // Identity element is 1
    // Inverse of 1 is 1, inverse of 4 is 4
    // Commutativity 1 * 4 == 4 * 1, 1 * 1 == 1 * 1
    // Hence (Z5*[2], *) is a commutative group
}

#[test]
fn exercise_40() {
    // Z6[6] = {0, 1, 2, 3, 4, 5}, whole group is a subgroup of itself
    // Z6[1] = {0}, neutral element is 0, inverse of 0 is 0, 0 + 0 == 0, trivial subgroup
    // Z6[2] = {0, 3}, neutral element is 0, inverse of 0 is 0, inverse of 3 is 3, 0 + 0 == 0, 3 + 3 == 0
    // Z6[3] = {0, 2, 4}, neutral element is 0, inverse of 0 is 0, inverse of 2 is 4, inverse of 4 is 2,
}

#[test]
fn exercise_41() {
    // if p >= 5 is a prime number, it is certainly odd number so p-1 is even number that group has a sugroup of order 2
    // Identity element for multiplication is 1, so in a subgroup of order 2 other element generates this subgroup, meaning
    // that it can not generate whole group
}

#[test]
fn exercise_42() {
    // e(g1*g1', g2) = e(g1, g2) * e(g1', g2)
    // g1' == g1 => e(g1^2, g2) = e(g1, g2) * e(g1, g2)
    // e(g1^a, g2) = e(g1, g2) * e(g1, g2) * ... * e(g1, g2) = e(g1, g2)^a
    // e(g1, g2^2) = e(g1, g2) * e(g1, g2)
    // e(g1, g2^b) = e(g1, g2) * e(g1, g2) * ... * e(g1, g2) = e(g1, g2)^b
    // e(g1^a, g2^b) = e(g1, g2)^a * e(g1, g2)^b = e(g1, g2)^(a+b)
}

#[test]
fn example_47() {
    use sha256::digest;

    let input = String::from("");
    let val = digest(input);
    assert_eq!(
        val,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[cfg(test)]
mod z13 {
    use ark_ff::fields::{Fp64, MontBackend, MontConfig};
    use ark_ff::{Field, PrimeField};

    #[derive(MontConfig)]
    #[modulus = "13"]
    #[generator = "1"]
    pub struct FqConfig;

    pub type Z13 = Fp64<MontBackend<FqConfig, 1>>;

    #[test]
    fn exercise_44() {
        fn pedersen(x: &[Z13], g: &[Z13]) -> Z13 {
            g.iter()
                .zip(x)
                .fold(Z13::from(1), |acc, (g, x)| acc * g.pow(x.into_bigint()))
        }

        let generators = [Z13::from(2), Z13::from(6), Z13::from(7)];
        let result = pedersen(&[Z13::from(3), Z13::from(7), Z13::from(11)], &generators);
        assert_eq!(result, Z13::from(8));
    }

    #[test]
    fn exercise_45() {
        fn pedersen_sha256(input: &str, g: &[Z13]) -> Z13 {
            use sha256::digest;
            let val = digest(input);
            let binary: Vec<u8> = hex::decode(&val).unwrap();
            let mut acc = Z13::from(1);
            for i in 0..g.len() {
                let b = (binary[i / 8] & (1 << (7 - i % 8))) >> 7 - i % 8;
                acc = acc * g[i].pow(&[b as u64]);
            }
            acc
        }

        let generators = [Z13::from(2), Z13::from(6), Z13::from(7)];
        let result = pedersen_sha256("TEST", &generators);
        assert_eq!(result, Z13::from(2));
    }
}

#[test]
fn exercise_47() {
    // (Z5, +) and (Z5 \0, *) are commutative groups with neutral elements 0 and 1 respectively
    // Characteristic of a Z5 field is 5, because adding 1 to itself 5 times gives 0
    // a*b = x - all the elements of field (Z5\0, *) are cofactors of 5, so they all have multiplicative inverse
    // It means that x = b * a^-1 has unique solutions
}

#[test]
fn exercise_48() {
    // (Z6 , +, ·) is not a field, because some elements (e.g. 2) have no multiplicative inverse
}

fn print_multiplication_table<T>()
where
    T: ark_ff::PrimeField + std::fmt::Display,
{
    let n: u64 = T::MODULUS.as_ref()[0];
    println!("Multiplication table for prime field {}", n);
    print!("  |");
    for i in 0..n {
        print!(" {:2}", i);
    }
    print!("\n-----------------------\n");
    for i in 0..n {
        print!("{:2} | ", i);
        for j in 0..n {
            let current = T::mul(T::from(i), T::from(j));
            // there is an issue with printing zero, so we cast to u64
            print!("{:2} ", current.into_bigint().as_ref()[0]);
        }
        println!();
    }
}

fn print_addition_table<T>()
where
    T: ark_ff::PrimeField + std::fmt::Display,
{
    let n: u64 = T::MODULUS.as_ref()[0];
    println!("Addition table for prime field {}", n);
    print!("  |");
    for i in 0..n {
        print!(" {:2}", i);
    }
    print!("\n------------\n");
    for i in 0..n {
        print!("{:2} | ", i);
        for j in 0..n {
            let current = T::add(T::from(i), T::from(j));
            // there is an issue with printing zero, so we cast to u64
            print!("{:2} ", current.into_bigint().as_ref()[0]);
        }
        println!();
    }
}

#[cfg(test)]
mod z3 {
    use ark_ff::fields::{Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "3"]
    #[generator = "1"]
    pub struct FqConfig;

    pub type Z3 = Fp64<MontBackend<FqConfig, 1>>;
    #[test]
    fn exercise_49() {
        crate::print_multiplication_table::<Z3>();
        crate::print_addition_table::<Z3>();
    }
}

#[test]
fn exercise_50() {
    crate::print_multiplication_table::<z13::Z13>();
    crate::print_addition_table::<z13::Z13>();
}

#[test]
fn exercise_51() {
    use ark_ff::PrimeField;
    use z13::Z13;
    let n: u64 = z13::Z13::MODULUS.as_ref()[0];
    let mut results: Vec<(u64, u64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let x = Z13::from(i);
            let y = Z13::from(j);
            if x.square() + y.square() == Z13::from(1) + Z13::from(7) * x.square() * y.square() {
                results.push((i, j));
            }
        }
    }
    println!("\nResults: {:?}", results);
}

#[test]
fn exercise_52() {
    use ark_ff::PrimeField;
    use z13::Z13;
    let p: u64 = z13::Z13::MODULUS.as_ref()[0];
    let mut results: Vec<(i8, Vec<Z13>)> = Vec::new();
    fn sqrt<T: PrimeField>(x: Z13) -> Option<(Z13, Z13)> {
        let p = T::MODULUS.as_ref()[0];
        for i in 0..p {
            let y = Z13::from(i);
            if y.square() == x {
                return Some((y, y.neg()));
            }
        }
        None
    }

    for i in 0..p {
        let x = Z13::from(i);
        let legrende = x.pow(&[(p - 1) / 2]);
        if legrende == Z13::from(1) {
            // Seems implementation does not work
            // let root = x.sqrt().unwrap();
            let roots = sqrt::<z13::Z13>(x).unwrap();
            results.push((1, vec![roots.0, roots.1]));
            println!("Checkpoint 3 {}", i);
        } else if legrende == Z13::from(p - 1) {
            results.push((-1, Vec::new()));
        } else {
            results.push((0, vec![Z13::from(0)]));
        }
    }
    println!("\nLegrende results for Z13: {:?}", results);
}

fn make_all_extension_field_polynomials<F: ark_ff::PrimeField, const D: usize>(
) -> Vec<ark_poly::univariate::SparsePolynomial<F>> {
    use itertools::Itertools;
    use std::collections::HashSet;
    let mut all_polynomials = Vec::new();
    let p: u64 = F::MODULUS.as_ref()[0];
    let degrees = [[0u64; D], [1u64; D]]
        .concat()
        .into_iter()
        .permutations(D)
        .unique();
    let coeffs = (0..p)
        .into_iter()
        .chain((0..p).into_iter())
        .into_iter()
        .permutations(D)
        .unique();

    let mut polynomials: HashSet<Vec<u64>> = HashSet::new();
    for degree in degrees {
        for coef in coeffs.clone() {
            let current_polynomial: Vec<u64> =
                degree.iter().zip(coef).map(|(d, c)| c * d).collect();
            polynomials.insert(current_polynomial);
        }
    }

    //println!("All the polynomials: {:?}", polynomials);

    for coeffs in polynomials {
        let terms = coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, c)| (D - i - 1, F::from(*c)))
            .collect();
        let pol = ark_poly::univariate::SparsePolynomial::from_coefficients_vec(terms);
        all_polynomials.push(pol);
    }

    all_polynomials
}

#[test]
fn exercise_53() {
    use ark_poly::univariate::SparsePolynomial;
    use z3::Z3;

    let prime_field_extension_values: Vec<SparsePolynomial<Z3>> =
        make_all_extension_field_polynomials::<Z3, 2usize>();

    let modulus = SparsePolynomial::from_coefficients_vec(vec![(2, Z3::from(1)), (0, Z3::from(1))]);

    for y in &prime_field_extension_values {
        for x in &prime_field_extension_values.clone() {
            let value = y.mul(y).add(
                x.mul(x)
                    .mul(x)
                    .add(SparsePolynomial::from_coefficients_vec(vec![(
                        0,
                        Z3::from(4),
                    )]))
                    .neg(),
            );
            if let Some((_q, r)) = DenseOrSparsePolynomial::from(&value)
                .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&modulus))
            {
                if r.is_zero() {
                    let y = if y.is_zero() {
                        "0".to_string()
                    } else {
                        format!("{:?}", y)
                    };
                    let x = if x.is_zero() {
                        "0".to_string()
                    } else {
                        format!("{:?}", x)
                    };
                    println!("Found solution: y = {}, x = \r{}\n", x, y);
                }
            } else {
                panic!("Division is not possible");
            }
        }
    }
}

#[test]
fn exercise_54() {
    use ark_poly::univariate::SparsePolynomial;
    use z3::Z3;

    let prime_field_extension_values: Vec<SparsePolynomial<Z3>> =
        make_all_extension_field_polynomials::<Z3, 2usize>();

    let modulus = SparsePolynomial::from_coefficients_vec(vec![
        (2, Z3::from(1)),
        (1, Z3::from(1)),
        (0, Z3::from(2)),
    ]);

    for v in 0..3 {
        if modulus.evaluate(&Z3::from(v)).is_zero() {
            panic!("Modulus is not irreducible");
        }
    }

    for x1 in &prime_field_extension_values {
        for x2 in &prime_field_extension_values {
            let value = x1.mul(&x2);
            if let Some((_q, r)) = DenseOrSparsePolynomial::from(&value)
                .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&modulus))
            {
                let x1 = if x1.is_zero() {
                    "0".to_string()
                } else {
                    format!("{:?}", x1)
                };
                let x2 = if x2.is_zero() {
                    "0".to_string()
                } else {
                    format!("{:?}", x2)
                };
                let r = if r.is_zero() {
                    "0".to_string()
                } else {
                    format!("{:?}", r)
                };
                println!("x1 = {}, x2 = {}, x1*x2= {}", x1, x2, r);
            }
        }
    }
}

mod z5 {
    use ark_ff::fields::{Fp64, MontBackend, MontConfig};

    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "1"]
    pub struct FqConfig;

    pub type Z5 = Fp64<MontBackend<crate::z5::FqConfig, 1>>;
}

#[test]
fn exercise_55() {
    use ark_poly::univariate::SparsePolynomial;
    use z5::Z5;

    let prime_field_extension_values: Vec<SparsePolynomial<Z5>> =
        make_all_extension_field_polynomials::<Z5, 3usize>();

    let modulus = SparsePolynomial::from_coefficients_vec(vec![
        (3, Z5::from(1)),
        (1, Z5::from(1)),
        (0, Z5::from(1)),
    ]);

    for v in 0..5 {
        if modulus.evaluate(&Z5::from(v)).is_zero() {
            panic!("Modulus is not irreducible value {}", v);
        }
    }

    let pol = SparsePolynomial::from_coefficients_vec(vec![(2, Z5::from(2)), (0, Z5::from(4))]);

    for pol2 in prime_field_extension_values {
        let value = pol.mul(&pol2);
        if let Some((_q, r)) = DenseOrSparsePolynomial::from(&value)
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&modulus))
        {
            if r == DenseOrSparsePolynomial::from(&SparsePolynomial::from_coefficients_vec(vec![(
                0,
                Z5::from(1),
            )]))
            .into()
            {
                println!("Found inverse: {:?}", pol2);
            }
        }
    }
    // (2*t^2 + 4)^-1 == 4t^2+4t+1
    // (2t^2 + 4)(x − (t^2 + 4t + 2)) = (2t + 3)
    // x - (t^2 + 4t + 2) = (2t + 3)*(4t^2+4t+1)
    // x  = (2t + 3)*(4t^2+4t+1) + (t^2 + 4t + 2)
    let pol1 = SparsePolynomial::from_coefficients_vec(vec![(1, Z5::from(2)), (0, Z5::from(3))]);
    let pol2 = SparsePolynomial::from_coefficients_vec(vec![
        (2, Z5::from(4)),
        (1, Z5::from(4)),
        (0, Z5::from(1)),
    ]);
    let pol3 = SparsePolynomial::from_coefficients_vec(vec![
        (2, Z5::from(1)),
        (1, Z5::from(4)),
        (0, Z5::from(2)),
    ]);
    let x = pol1.mul(&pol2) + pol3;
    if let Some((_q, r)) = DenseOrSparsePolynomial::from(&x)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&modulus))
    {
        println!("x = {:?}", r);
    }
}

#[test]
fn exercise_56() {
    use ark_poly::univariate::SparsePolynomial;
    use z5::Z5;

    let modulus = SparsePolynomial::from_coefficients_vec(vec![(2, Z5::from(1)), (0, Z5::from(2))]);

    for v in 0..5 {
        if modulus.evaluate(&Z5::from(v)).is_zero() {
            panic!("Modulus is not irreducible value {}", v,);
        }
    }

    let prime_field_extension_values: Vec<SparsePolynomial<Z5>> =
        make_all_extension_field_polynomials::<Z5, 2usize>();

    println!("All F5[2] polynomials: {:?}", prime_field_extension_values);

}
