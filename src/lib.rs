

//use ndarray::prelude::*;

use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

pub fn naive_dot<T>(x0 : &[T], x1 : &[T]) -> T
where T : Add<Output=T>+Mul<Output=T>+Sub<Output=T>+From<f64> + Copy{
    let zero = T::from(0.0);
    let out = x0.iter().zip(x1.iter()).fold(zero,
    |acc,y|{
        let (y0,y1) = y;
        let product = (*y0)*(*y1);
        acc+product
    });
    out
}


pub fn dot<T>(x0 : &[T], x1 : &[T]) -> T
where T : Add<Output=T>+Mul<Output=T>+Sub<Output=T>+From<f64> + Copy{
    let zero = T::from(0.0);
    let (out,_compensate) = x0.iter().zip(x1.iter()).fold((zero,zero),
    |acc,y|{
        let (dot,compensate) = acc; 
        let (y0,y1) = y;
        let product = (*y0)*(*y1);
        let z = product - compensate;
        let t = dot + z;
        (t,(t-dot)-z)
    });
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::{StdRng};
    use rand::{Rng,SeedableRng};
    use float_cmp::ApproxEq;



    #[test]
    fn simple_dot_test() {
        let x1 = vec![1.0,2.0,3.0];
        let x2 = vec![1.0,1.0/2.0,1.0/3.0];
        assert_eq!(dot(&x1,&x2),3.0);
    }

    #[test]
    fn accuracy_dot_test() {
        let nsizes=10000;
        for n in 1..nsizes{
            let x1 = vec![0.123;n];
            let x2 = vec![1.0;n];
            let pass = dot(&x1,&x2).approx_eq((n as f64)*0.123,(0.0,10));
            if !pass{
                println!("FAILED RESULTS:\n{}\n{}\n",dot(&x1,&x2),(n as f64)*0.123);
            }
            assert!(pass);
        }
    }



    #[test]
    fn uniformly_distributed_right_additive_dot_test(){
        let seed = 98712983 as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        let nsizes=1000;
        for n in 1..nsizes{

            let x : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();
            let y : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();
            let z : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();

            let ypz : Vec<_> = y.iter().zip(z.iter()).map(|yz|{
                let (u,v)=yz;
                u+v
            }).collect();


            let left = dot(&x,&ypz);
            let right = dot(&x,&y)+dot(&x,&z);
            let left_equals_right = left.approx_eq(right,(0.0,10));
            if !left_equals_right{
                println!("{}\n{}\n",left,right);
            }
            assert!(left_equals_right);
        }
    }

    #[test]
    fn uniformly_distributed_left_multiplicative_test(){
        let seed = 98712983 as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        let nsizes=1000;
        for n in 1..nsizes{

            let x : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();
            let y : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();

            let a = rng.gen_range(-10.0,10.0);
            let ax : Vec<_> = x.iter().map(|z|{
                a*z
            }).collect();


            let left = dot(&ax,&y);
            let right = a*dot(&x,&y);
            let left_equals_right = left.approx_eq(right,(0.0,20));
            if !left_equals_right{
                println!("{}\n{}\n",left,right);
            }
            assert!(left_equals_right);
        }
    }

    #[test]
    fn uniformly_distributed_right_multiplicative_test(){
        let seed = 98712983 as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        let nsizes=1000;
        for n in 1..nsizes{

            let x : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();
            let y : Vec<_> = (0..n).map(|_|{rng.gen_range(-1.0f64,2.0f64)}).collect();

            let a = rng.gen_range(-10.0,10.0);
            let ax : Vec<_> = x.iter().map(|z|{
                a*z
            }).collect();


            let left = dot(&y,&ax);
            let right = a*dot(&y,&x);
            let left_equals_right = left.approx_eq(right,(0.0,20));
            if !left_equals_right{
                println!("{}\n{}\n",left,right);
            }
            assert!(left_equals_right);
        }
    }



}