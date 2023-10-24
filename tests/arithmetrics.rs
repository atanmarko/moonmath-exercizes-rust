// Various rust crates are used to solve the exercises (for practise and demonstration purposes)

#[test]
fn exercise_1() {
    println!(
        "Absolute values of integers -123, 27, and 0 are: {}, {}, and {}",
        (-123i32).abs(),
        27_i32.abs(),
        0i32.abs()
    );
}

#[test]
fn exercise_2() {
    let mut num = 30030;
    let factorials: Vec<_> = (2..num).filter(|&x| num % x == 0).collect();
    println!("All the factorials of {} are: {:?}", num, factorials);

    let mut min_factorials: Vec<i32> = Vec::new();

    for factor in factorials {
        if num % factor == 0 {
            min_factorials.push(factor);
            num /= factor;
            if num == 1 {
                break;
            }
        }
    }

    println!(
        "The minimum factorials of {} are: {:?}",
        num, min_factorials
    );
}

#[test]
fn exercise_3() {
    // 4 · x + 21 = 5
    use roots::{find_roots_linear, Roots};
    let roots = find_roots_linear::<f32>(4., 21. - 5.);
    println!("Roots of 4x + 21 = 5 are: {:?}", roots);

    match roots {
        Roots::No(_) => {
            println!("No roots found");
        }
        Roots::One(values) => {
            let solution_integers = values
                .iter()
                .filter(|r| (**r).fract() == 0.0)
                .map(|&r| r as i32)
                .collect::<Vec<i32>>();
            let solution_natural_numbers = solution_integers
                .iter()
                .filter(|&&i| i > 0)
                .collect::<Vec<_>>();
            println!(
                "Solution integers {:?}, solution natural numbers: {:?} ",
                solution_integers, solution_natural_numbers
            );
        }
        _ => {
            println!("Two roots found, that could not happen for degree 1 equation");
        }
    }
}

#[test]
fn exercise_4() {
    use roots::find_roots_cubic;
    // 2*x^3 - x^2 - 2x + 1 = 0
    use roots::Roots;
    let roots = find_roots_cubic::<f32>(2., -1., -2., 1.);

    match roots {
        Roots::No(_) => {
            println!("No roots found");
        }
        Roots::Three(values) => {
            let solution_rational_numbers = values;
            let solution_integers = values
                .iter()
                .filter(|r| (**r).fract() == 0.0)
                .map(|&r| r as i32)
                .collect::<Vec<i32>>();
            let solution_natural_numbers = solution_integers
                .iter()
                .filter(|&&i| i > 0)
                .collect::<Vec<_>>();
            println!(
                "Solution integers {:?}, solution natural numbers: {:?} solution rational numbers: {:.3?}",
                solution_integers, solution_natural_numbers, solution_rational_numbers
            );
        }
        _ => {
            println!("That could not happen for degree 3 equation");
        }
    }
}

#[test]
fn exercise_5() {
    let find_mr = |(a, b): (i32, i32)| {
        for r in 0..b.abs() {
            let m: f32 = (a - r) as f32 / b as f32;
            if m.fract() == 0.0 {
                println!("Case ({},{}) solution exists, m={}, r={}", a, b, m, r);
                return;
            }
        }
        println!("Case ({},{}) solution does not exist", a, b);
    };

    find_mr((27, 5));
    find_mr((-27, 5));
    find_mr((127, 0));
    find_mr((-1687, 11));
    find_mr((0, 7));
}

#[test]
fn exercise_6() {
    // Implement long division in Rust
    // a = b*q + r, r>=0, r<b
    fn long_division(a: i32, b: i32) -> (i32 /*q*/, i32 /*r*/) {
        assert_ne!(b, 0);
        let mut q = 0;

        while (a - (q * b)).abs() >= b.abs() {
            if (a < 0 && b > 0) || (a > 0 && b < 0) {
                q -= 1;
            } else {
                q += 1;
            }
        }

        let mut r = (a - q * b).abs();
        if a < 0 && b < 0 {
            q += 1;
            r = (a - q * b).abs();
        }
        (q, r)
    }

    assert_eq!(long_division(27, 5), (5, 2));
    assert_eq!(long_division(27, -5), (-5, 2));
    assert_eq!(long_division(143785, 17), (8457, 16));
    assert_eq!(long_division(-17, -4), (5, 3));
    assert_eq!(long_division(-3, -2), (2, 1));
}

#[test]
fn exercise_7() {
    fn binary_representation(mut number: u32) -> String {
        let mut result = String::with_capacity(32);

        while number > 0 {
            result.push_str(if number % 2 == 0 { "0" } else { "1" });
            number /= 2;
        }
        result.chars().rev().collect::<String>()
    }

    assert_eq!(binary_representation(12), "1100");
    assert_eq!(binary_representation(2), "10");
    assert_eq!(binary_representation(31), "11111");
    assert_eq!(binary_representation(17), "10001");
}

// Extended euclidean algorithm
// gcd(a,b) = s*a + t*b
// Returns (gcd, s, t)
fn egcd(a: i32, b: i32) -> (i32, i32, i32) {
    let mut r: Vec<i32> = vec![0i32; 64];
    let mut s: Vec<i32> = vec![0i32; 64];
    let mut t: Vec<i32> = vec![0i32; 64];
    let mut k = 2usize;

    r[0] = a;
    r[1] = b;
    s[0] = 1;
    s[1] = 0;
    t[0] = 0;
    t[1] = 1;

    while r[k - 1] != 0 {
        let q = r[k - 2] / r[k - 1];
        r[k] = r[k - 2] % r[k - 1];
        s[k] = s[k - 2] - q * s[k - 1];
        t[k] = t[k - 2] - q * t[k - 1];
        k += 1;
    }
    (r[k - 2], s[k - 2], t[k - 2])
}

#[test]
fn exercise_8() {
    let (gcd, s, t) = egcd(45, 10);
    println!("gcd(45,10) = {} = {}*45 + {}*10", gcd, s, t);
    let (gcd, s, t) = egcd(13, 11);
    println!("gcd(13,11) = {} = {}*13 + {}*11", gcd, s, t);
    let (gcd, s, t) = egcd(13, 12);
    println!("gcd(13,12) = {} = {}*13 + {}*12", gcd, s, t);
}

#[test]
fn exercise_9() {
    let prime: i32 = num_primes::Generator::new_prime(31).to_u32_digits()[0] as i32;
    assert!(num_primes::Verification::is_prime(&(prime as u32).into()));
    let n: i32 = num_primes::Generator::new_uint(16).to_u32_digits()[0] as i32;
    let (gcd, s, t) = egcd(prime, n);
    println!(
        "gcd({},{}) = {} = {}*{} + {}*{}",
        prime, n, gcd, s, prime, t, n
    );

    assert_eq!(gcd, 1);
}

#[test]
fn exercise_10() {
    let results = (0..=100)
        .step_by(1)
        .filter(|k| egcd(100, *k).0 == 5)
        .collect::<Vec<_>>();
    println!(
        "all numbers k ∈ N with 0 ≤ k ≤ 100 such that gcd(100, k) are: {:?}",
        results
    );
}

#[test]
fn exercise_12() {
    let pairs = vec![(45, 10), (13, 11), (13, 12)];
    for (a, b) in pairs {
        let (gcd, _, _) = egcd(a, b);
        if gcd == 1 {
            println!("Numbers {} and {} are coprime", a, b);
        }
    }
}

#[test]
fn exercise_13() {
    let num1_octal = vec![1, 3, 5, 4];
    let num1_decimal = num1_octal
        .iter()
        .rev()
        .enumerate()
        .fold(0, |acc, (i, &x)| acc + x * 8i32.pow(i as u32));
    println!(
        "Decimal representation of {:?} is {}",
        num1_octal, num1_decimal
    );

    let num2_octal = vec![7, 7, 7];
    let num2_decimal = num2_octal
        .iter()
        .rev()
        .enumerate()
        .fold(0, |acc, (i, &x)| acc + x * 8i32.pow(i as u32));
    println!(
        "Decimal representation of {:?} is {}",
        num2_octal, num2_decimal
    );
}

fn is_congruent(a: i32, b: i32, n: i32) -> bool {
    (a - b) % n == 0
}

#[test]
fn exercise_14() {
    println!(
        "Is 5 congruent to 19 modulo 13? {}",
        is_congruent(5, 19, 13)
    );
    println!(
        "Is 13 congruent to 0 modulo 13? {}",
        is_congruent(13, 0, 13)
    );
    println!(
        "Is -4 congruent to 9 modulo 13? {}",
        is_congruent(-4, 9, 13)
    );
    println!("Is 0 congruent to 0 modulo 13? {}", is_congruent(0, 0, 13));
}

#[test]
fn exercise_15() {
    let congruent_to_4_mod_6 = (0..=100)
        .step_by(1)
        .filter(|&x| is_congruent(x, 4, 6))
        .collect::<Vec<_>>();

    println!(
        "All numbers congruent to 4 modulo 6 are: {:?};\n All numbers in the form i*6+4, i>0",
        congruent_to_4_mod_6
    );
}

#[test]
fn exercise_16() {
    // 5x + 4 == 28 + 2x (mod 13)
    // 5x + 4 - 4 - 2x = 28 - 4 + 2x - 2x (mod 13)
    // 3x = 24 (mod 13)
    //  k^(p−1) ≡ 1 ( mod p ) => 3^(13−2) ≡ 1/3 ( mod 13 )
    // 3x * 3^11 = 24 * 3 ^11 (mod 13)
    // 3^11 mod 13 = 9
    // x = 24 * 9 (mod 13)
    // x = 216 (mod 13)
    // x = 8 (mod 13)
    let x = 8;
    assert_eq!((5 * x + 4) % 13, (28 + 2 * x) % 13);
}

#[test]
fn exercise_17() {
    use rand::Rng;
    // 69 * x = 5 (mod 23)
    // 0 * x = 5, no solution

    let x = rand::thread_rng().gen_range(1..100);
    assert_ne!((69 * x) % 23, 5);
}

#[test]
fn exercise_18() {
    use rand::Rng;
    // 69 * x = 46 (mod 23)
    // 0 * x = 0, every x is solution

    let x = rand::thread_rng().gen_range(1..100);
    assert_eq!((69 * x) % 23, 46 % 23);
}

#[test]
fn exercise_21() {
    use rand::Rng;
    // 5x + 4 ≡ 28 + 2x ( mod 13 )
    // 5x + 4 = 2 + 2x
    // 5x + 4 - 2x -4 = 2 + 2x - 2x -4
    // 3x = -2
    // 3x = 11
    // 3x * 9 = 11 * 9
    // 27x = 99
    // x = 8
    // x = 13 * k + 8
    let k = rand::thread_rng().gen_range(-100..100);
    let x = 13 * k + 8;
    assert_eq!((5 * x + 4) % 13, (28 + 2 * x) % 13);
}

#[test]
fn exercise_22() {
    let numbers = [7, 1, 0, 805, -4255]
        .map(|n| n % 24)
        .map(|n| if n < 0 { 24 + n } else { n });
    let modulus = 24;
    for n in numbers {
        if n == 1 {
            println!("Number 1 is trivially always its own inverse");
            continue;
        }
        let (gcd, _s, t) = egcd(modulus, n);
        if gcd == 1 {
            let inv = match t % modulus {
                inv if inv < 0 => 24 + inv,
                inv => inv,
            };
            println!(
                "Number {:?} has multiplicative inverse {:?} in modulo 24 arithmetic",
                n, inv
            );
        } else {
            println!(
                "Number {:?} does not have inverse in modulo 24 arithmetic",
                n
            );
        }
    }
}

#[test]
fn exercise_23() {
    use rand::Rng;
    // 17(2x + 5) − 4 ≡ 2x + 4 ( mod 5 )
    // 2*(2x) + 1 ≡ 2x + 4 ( mod 5 )
    // 4x + 1 ≡ 2x + 4 ( mod 5 )
    // 4x + 1 - 2x - 1 ≡ 2x - 2x + 4 - 1 ( mod 5 )
    // 2x ≡ 3 ( mod 5 ), 2^-1 = 2^(5-2) = 8 ( mod 5 ) = 3
    // 2x * 3 ≡ 3 * 3 ( mod 5 )
    // x ≡ 4

    let k = rand::thread_rng().gen_range(-100..100);
    let x = 5 * k + 4;
    assert_eq!((17 * (2 * x + 5) - 4) % 5, (2 * x + 4) % 5);
}

#[test]
fn exercise_24() {
    // 17(2x + 5) − 4 ≡ 2x + 4 ( mod 6 )
    // 34x +  85 - 4 ≡ 2x + 4 ( mod 6 )
    // 34x +  81 - 81 -2x ≡ 2x + 4 - 2x - 81 ( mod 6 )
    // 32x ≡ -77 ( mod 6 )
    // 2x ≡ 1 ( mod 6 )
    // No solution

    for x in -100..100 {
        assert_ne!((17 * (2 * x + 5) - 4) % 6, (2 * x + 4) % 6);
    }
}

#[test]
fn exercise_25() {
    // P7 ∈ Z[x], P7(x) = (x - 2) * (x + 3) * (x - 5)
    // P7(x) = x^3 - 4x^2 - 11x + 30
    // P7 ∈ Z6[x], P7(x) = x^3 + 2*x^2 + x
}

#[test]
fn exercise_26() {
    // (P + Q)(x) = (5x2 − 4x + 2) + (x3 − 2x2 + 5) = x3 + 3x2 + 2x + 1
    // (P · Q)(x) = (5x2 − 4x + 2) · (x3 − 2x2 + 5) = 5x5 + 4x4 + 4x3 + 3x2 + 4x + 4
    // We can do the projection of result from Z[x] to Z6[x] by taking modulo 6 of each coefficient
    // In this case (P + Q)(x) and (P · Q)(x)  are the same in Z6[x] and Z[x]
}

#[cfg(test)]
mod z5 {
    use ark_ff::fields::{Fp64, MontBackend, MontConfig};
    use ark_poly::polynomial::univariate::{DenseOrSparsePolynomial, SparsePolynomial};
    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "1"]
    pub struct FqConfig;
    pub type Z5 = Fp64<MontBackend<FqConfig, 1>>;

    #[test]
    fn exercise_27() {
        // A(x) := −3x^4 + 4x^3 + 2x^2 + 4, B(x) =x^2 − 4x + 2; A, B ∈ Z[x]
        // A(x) := 2x^4 + 4x^3 + 2x^2 + 4, B(x) =x^2 + 1x + 2; A, B ∈ Z5[x]
        let a = SparsePolynomial::from_coefficients_vec(vec![
            (4, Z5::from(2)),
            (3, Z5::from(4)),
            (2, Z5::from(2)),
            (0, Z5::from(4)),
        ]);
        let b = SparsePolynomial::from_coefficients_vec(vec![
            (2, Z5::from(1)),
            (1, Z5::from(1)),
            (0, Z5::from(2)),
        ]);

        if let Some((q, r)) = DenseOrSparsePolynomial::from(&a)
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&b))
        {
            println!("Q(x) = {:?},\n R(x) = {:?}", q, r);
        } else {
            panic!("Division is not possible");
        }
    }

    #[test]
    fn exercise_28() {
        // A(X) = x^7 + 4*x^6 + 4*x^5 + x^3 + 2*x^2 + 2x + 3, B(x) := 2x^4 +2x + 4; A, B ∈ Z5[x]
        let a = SparsePolynomial::from_coefficients_vec(vec![
            (7, Z5::from(1)),
            (6, Z5::from(4)),
            (5, Z5::from(4)),
            (3, Z5::from(1)),
            (2, Z5::from(2)),
            (1, Z5::from(2)),
            (0, Z5::from(3)),
        ]);

        let b = SparsePolynomial::from_coefficients_vec(vec![
            (4, Z5::from(2)),
            (1, Z5::from(2)),
            (0, Z5::from(4)),
        ]);

        if let Some((q, r)) = DenseOrSparsePolynomial::from(&a)
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&b))
        {
            println!("Q(x) = {:?},\n R(x) = {:?}", q, r);
            assert_eq!(
                r,
                SparsePolynomial::from_coefficients_vec(vec![(0, Z5::from(0))]).into()
            );
        } else {
            panic!("Division is not possible");
        }
    }

    fn lagrange_interpolation(points: Vec<(Z5, Z5)>) -> SparsePolynomial<Z5> {
        use ark_ff::Field;
        let mut result = SparsePolynomial::from_coefficients_vec(vec![(0, Z5::from(0))]);
        for (i, (x_i, y_i)) in points.iter().enumerate() {
            let mut numerator = SparsePolynomial::from_coefficients_vec(vec![(0, *y_i)]);
            let mut denominator = Z5::from(1);
            for (j, (x_j, _)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                numerator = numerator.mul(&SparsePolynomial::from_coefficients_vec(vec![
                    (0, Z5::from(0) - x_j),
                    (1, Z5::from(1)),
                ]));
                denominator *= x_i - x_j;
            }
            numerator = numerator.mul(&SparsePolynomial::from_coefficients_vec(vec![(
                0,
                denominator.inverse().unwrap(),
            )]));
            result = result + numerator;
        }
        result
    }

    #[test]
    fn exercise_31() {
        // S = {(0, 0), (1, 1), (2, 2), (3, 2)}
        let points = vec![
            (Z5::from(0), Z5::from(0)),
            (Z5::from(1), Z5::from(1)),
            (Z5::from(2), Z5::from(2)),
            (Z5::from(3), Z5::from(2)),
        ];
        let result = lagrange_interpolation(points);
        println!("Lagrange interpolation result: {:?}", result);
    }

    #[test]
    fn exercise_32() {
        // S = {(0, 0), (1, 1), (2, 2), (3, 2)}
        // In Z6, some elements do not have multiplicative inverses so we cannot perform lagrange interpolation
    }
}
