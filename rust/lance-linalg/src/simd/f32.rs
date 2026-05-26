// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `f32x8`, 8 of `f32` values.s

use std::fmt::Formatter;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "loongarch64")]
use std::arch::loongarch64::*;
#[cfg(target_arch = "powerpc64")]
use std::arch::powerpc64::*;
use std::mem::transmute;
#[cfg(target_arch = "powerpc64")]
#[inline(always)]
unsafe fn power10_f32_min(a: vector_float, b: vector_float) -> vector_float {
    let mut mask: vector_float;
    std::arch::asm!(
        "xvcmpgtsp {0}, {1}, {2}",
        out(vsreg) mask,
        in(vsreg) b,
        in(vsreg) a
    );
    vec_sel(b, a, std::mem::transmute::<vector_float, vector_unsigned_int>(mask))
}
#[inline]
#[target_feature(enable = "altivec")]
unsafe fn powerpc_f32_cmpeq(a: vector_float, b: vector_float) -> vector_unsigned_int {
    let mut res: vector_unsigned_int;
    std::arch::asm!(
        "xvcmpeqsp {0}, {1}, {2}",
        out(vsreg) res,
        in(vsreg) a,
        in(vsreg) b
    );
    res
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "loongarch64")]
use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::{FloatSimd, SIMD};

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct f32x8(std::arch::x86_64::__m256);

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct f32x8(float32x4x2_t);

/// 8 of 32-bit `f32` values. Use 256-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "loongarch64")]
#[derive(Clone, Copy)]
pub struct f32x8(v8f32);

/// 8 of 32-bit `f32` values.
#[allow(non_camel_case_types)]
#[cfg(any(target_arch = "powerpc64"))]
#[derive(Clone, Copy)]
pub struct f32x8(pub vector_float, pub vector_float);

impl std::fmt::Debug for f32x8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0_f32; 8];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "f32x8({:?})", arr)
    }
}

impl f32x8 {
    #[inline]
    pub fn gather(slice: &[f32], indices: &[i32; 8]) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use super::i32::i32x8;

            let idx = i32x8::from(indices);
            Self(_mm256_i32gather_ps::<4>(slice.as_ptr(), idx.0))
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            // aarch64 does not have relevant SIMD instructions.
            let ptr = slice.as_ptr();

            let values = [
                *ptr.add(indices[0] as usize),
                *ptr.add(indices[1] as usize),
                *ptr.add(indices[2] as usize),
                *ptr.add(indices[3] as usize),
                *ptr.add(indices[4] as usize),
                *ptr.add(indices[5] as usize),
                *ptr.add(indices[6] as usize),
                *ptr.add(indices[7] as usize),
            ];
            Self::load_unaligned(values.as_ptr())
        }

        #[cfg(target_arch = "loongarch64")]
        unsafe {
            // loongarch64 does not have relevant SIMD instructions.
            let ptr = slice.as_ptr();

            let values = [
                *ptr.add(indices[0] as usize),
                *ptr.add(indices[1] as usize),
                *ptr.add(indices[2] as usize),
                *ptr.add(indices[3] as usize),
                *ptr.add(indices[4] as usize),
                *ptr.add(indices[5] as usize),
                *ptr.add(indices[6] as usize),
                *ptr.add(indices[7] as usize),
            ];
            Self::load_unaligned(values.as_ptr())
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            // PowerPC does not have a native gather instruction for f32x8.
            let ptr = slice.as_ptr();
            let values = [
                *ptr.add(indices[0] as usize),
                *ptr.add(indices[1] as usize),
                *ptr.add(indices[2] as usize),
                *ptr.add(indices[3] as usize),
                *ptr.add(indices[4] as usize),
                *ptr.add(indices[5] as usize),
                *ptr.add(indices[6] as usize),
                *ptr.add(indices[7] as usize),
            ];
            Self::load_unaligned(values.as_ptr())
        }
    }
}

impl From<&[f32]> for f32x8 {
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 8]> for f32x8 {
    fn from(value: &'a [f32; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 8> for f32x8 {
    fn splat(val: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(vdupq_n_f32(val), vdupq_n_f32(val)))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(transmute(lasx_xvreplgr2vr_w(transmute(val))))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(vec_splats(val), vec_splats(val))
        }
    }

    fn zeros() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_setzero_ps())
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::splat(0.0)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self::splat(0.0)
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            Self::splat(0.0)
        }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_load_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::load_unaligned(ptr)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self(transmute(lasx_xvld::<0>(transmute(ptr))))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            Self::load_unaligned(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self(vld1q_f32_x2(ptr))
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self(transmute(lasx_xvld::<0>(transmute(ptr))))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            Self(vec_xl(0, ptr), vec_xl(16, ptr))
        }
    }

    unsafe fn store(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_store_ps(ptr, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x2(ptr, self.0);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            lasx_xvst::<0>(transmute(self.0), transmute(ptr));
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.store_unaligned(ptr);
        }
    }

    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_ps(ptr, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x2(ptr, self.0);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            lasx_xvst::<0>(transmute(self.0), transmute(ptr));
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            vec_xst(self.0, 0, ptr);
            vec_xst(self.1, 16, ptr);
        }
    }

    #[inline]
    fn reduce_sum(&self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut sum = self.0;
            // Shift and add vector, until only 1 value left.
            // sums = [x0-x7], shift = [x4-x7]
            let mut shift = _mm256_permute2f128_ps(sum, sum, 1);
            // [x0+x4, x1+x5, ..]
            sum = _mm256_add_ps(sum, shift);
            shift = _mm256_permute_ps(sum, 14);
            sum = _mm256_add_ps(sum, shift);
            sum = _mm256_hadd_ps(sum, sum);
            let mut results: [f32; 8] = [0f32; 8];
            _mm256_storeu_ps(results.as_mut_ptr(), sum);
            results[0]
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let sum = vaddq_f32(self.0.0, self.0.1);
            vaddvq_f32(sum)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            self.as_array().iter().sum()
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            self.as_array().iter().sum()
        }
    }

    fn reduce_min(&self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                let mut min = self.0;
                // Shift and add vector, until only 1 value left.
                // sums = [x0-x7], shift = [x4-x7]
                let mut shift = _mm256_permute2f128_ps(min, min, 1);
                // [x0+x4, x1+x5, ..]
                min = _mm256_min_ps(min, shift);
                shift = _mm256_permute_ps(min, 14);
                min = _mm256_min_ps(min, shift);
                shift = _mm256_permute_ps(min, 1);
                min = _mm256_min_ps(min, shift);
                _mm256_cvtss_f32(min)
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let m = vminq_f32(self.0.0, self.0.1);
            vminvq_f32(m)
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            let m1 = lasx_xvpermi_d::<14>(transmute(self.0));
            let m2 = lasx_xvfmin_s(transmute(m1), self.0);
            let m1 = lasx_xvpermi_w::<14>(transmute(m2), transmute(m2));
            let m2 = lasx_xvfmin_s(transmute(m1), transmute(m2));
            let m1 = lasx_xvpermi_w::<1>(transmute(m2), transmute(m2));
            let m2 = lasx_xvfmin_s(transmute(m1), transmute(m2));
            transmute(lasx_xvpickve2gr_w::<0>(transmute(m2)))
        }
        #[cfg(target_arch = "powerpc64")]
        unsafe {
            // Step 1: Compare the two halves of f32x8 to get a single vector_float
            let v = power10_f32_min(self.0, self.1);

            // Step 2: Horizontal reduction using the ASM you just verified
            let mut v_shift8: vector_float;
            std::arch::asm!(
                "xxsldwi {0}, {1}, {1}, 2",
                out(vsreg) v_shift8,
                in(vsreg) v
            );
            let v_min2 = power10_f32_min(v, v_shift8);

            let mut v_shift4: vector_float;
            std::arch::asm!(
                "xxsldwi {0}, {1}, {1}, 1",
                out(vsreg) v_shift4,
                in(vsreg) v_min2
            );
            let v_min1 = power10_f32_min(v_min2, v_shift4);

            // Step 3: Extract the result from the first lane
            let res: [f32; 4] = std::mem::transmute(v_min1);
            res[0]
        }
    }

    fn min(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_min_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vminq_f32(self.0.0, rhs.0.0),
                vminq_f32(self.0.1, rhs.0.1),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfmin_s(self.0, rhs.0))
        }
        #[cfg(target_arch = "powerpc64")]
        unsafe {
            Self(
                power10_f32_min(self.0, rhs.0),
                power10_f32_min(self.1, rhs.1),
            )
        }
    }

    fn find(&self, val: f32) -> Option<i32> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            for i in 0..8 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let tgt = vdupq_n_f32(val);
            let mut arr = [0; 8];
            let mask1 = vceqq_f32(self.0.0, tgt);
            let mask2 = vceqq_f32(self.0.1, tgt);
            vst1q_u32(arr.as_mut_ptr(), mask1);
            vst1q_u32(arr.as_mut_ptr().add(4), mask2);
            for i in 0..8 {
                if arr.get_unchecked(i) != &0 {
                    return Some(i as i32);
                }
            }
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            for i in 0..8 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            for i in 0..8 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
        }
        None
    }
}

impl FloatSimd<f32, 8> for f32x8 {
    fn multiply_add(&mut self, a: Self, b: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_fmadd_ps(a.0, b.0, self.0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vfmaq_f32(self.0.0, a.0.0, b.0.0);
            self.0.1 = vfmaq_f32(self.0.1, a.0.1, b.0.1);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfmadd_s(a.0, b.0, self.0);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_madd(a.0, b.0, self.0);
            self.1 = vec_madd(a.1, b.1, self.1);
        }
    }
}

impl Add for f32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_add_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vaddq_f32(self.0.0, rhs.0.0),
                vaddq_f32(self.0.1, rhs.0.1),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfadd_s(self.0, rhs.0))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_add(self.0, rhs.0),
                vec_add(self.1, rhs.1),
            )
        }
    }
}

impl AddAssign for f32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_add_ps(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vaddq_f32(self.0.0, rhs.0.0);
            self.0.1 = vaddq_f32(self.0.1, rhs.0.1);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfadd_s(self.0, rhs.0);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_add(self.0, rhs.0);
            self.1 = vec_add(self.1, rhs.1);
        }
    }
}

impl Sub for f32x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_sub_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vsubq_f32(self.0.0, rhs.0.0),
                vsubq_f32(self.0.1, rhs.0.1),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfsub_s(self.0, rhs.0))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_sub(self.0, rhs.0),
                vec_sub(self.1, rhs.1),
            )
        }
    }
}

impl SubAssign for f32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.0 = _mm256_sub_ps(self.0, rhs.0)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vsubq_f32(self.0.0, rhs.0.0);
            self.0.1 = vsubq_f32(self.0.1, rhs.0.1);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfsub_s(self.0, rhs.0);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_sub(self.0, rhs.0);
            self.1 = vec_sub(self.1, rhs.1);
        }
    }
}

impl Mul for f32x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            Self(_mm256_mul_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x2_t(
                vmulq_f32(self.0.0, rhs.0.0),
                vmulq_f32(self.0.1, rhs.0.1),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfmul_s(self.0, rhs.0))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_mul(self.0, rhs.0),
                vec_mul(self.1, rhs.1),
            )
        }
    }
}

/// 16 of 32-bit `f32` values. Use 512-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
#[derive(Clone, Copy)]
pub struct f32x16(__m256, __m256);
#[allow(non_camel_case_types)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[derive(Clone, Copy)]
pub struct f32x16(__m512);

/// 16 of 32-bit `f32` values. Use 512-bit SIMD if possible.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct f32x16(float32x4x4_t);

/// 16 of 32-bit `f32` values. Use 256-bit SIMD
#[allow(non_camel_case_types)]
#[cfg(target_arch = "loongarch64")]
#[derive(Clone, Copy)]
pub struct f32x16(v8f32, v8f32);

#[allow(non_camel_case_types)]
#[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
#[derive(Clone, Copy)]
pub struct f32x16(pub(crate) vector_float, pub(crate) vector_float, pub(crate) vector_float, pub(crate) vector_float);

impl std::fmt::Debug for f32x16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arr = [0.0_f32; 16];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        write!(f, "f32x16({:?})", arr)
    }
}

impl From<&[f32]> for f32x16 {
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 16]> for f32x16 {
    fn from(value: &'a [f32; 16]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 16> for f32x16 {
    #[inline]
    fn splat(val: f32) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_set1_ps(val))
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_set1_ps(val), _mm256_set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vdupq_n_f32(val),
                vdupq_n_f32(val),
                vdupq_n_f32(val),
                vdupq_n_f32(val),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(
                transmute(lasx_xvreplgr2vr_w(transmute(val))),
                transmute(lasx_xvreplgr2vr_w(transmute(val))),
            )
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            let v = vec_splats(val);
            Self(v, v, v, v)
        }
    }

    #[inline]
    fn zeros() -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_setzero_ps())
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_setzero_ps(), _mm256_setzero_ps())
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::splat(0.0)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self::splat(0.0)
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            Self::splat(0.0)
        }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_load_ps(ptr), _mm256_load_ps(ptr.add(8)))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_load_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::load_unaligned(ptr)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self(
                transmute(lasx_xvld::<0>(transmute(ptr))),
                transmute(lasx_xvld::<32>(transmute(ptr))),
            )
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_xl(0, ptr),
                vec_xl(16, ptr),
                vec_xl(32, ptr),
                vec_xl(48, ptr),
            )
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr.add(8)))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self(vld1q_f32_x4(ptr))
        }
        #[cfg(target_arch = "loongarch64")]
        {
            Self(
                transmute(lasx_xvld::<0>(transmute(ptr))),
                transmute(lasx_xvld::<32>(transmute(ptr))),
            )
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_xl(0, ptr),
                vec_xl(16, ptr),
                vec_xl(32, ptr),
                vec_xl(48, ptr),
            )
        }
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f32) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            _mm512_store_ps(ptr, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            _mm256_store_ps(ptr, self.0);
            _mm256_store_ps(ptr.add(8), self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x4(ptr, self.0);
        }
        #[cfg(target_arch = "loongarch64")]
        {
            lasx_xvst::<0>(transmute(self.0), transmute(ptr));
            lasx_xvst::<32>(transmute(self.1), transmute(ptr));
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            vec_xst(self.0, 0, ptr);
            vec_xst(self.1, 16, ptr);
            vec_xst(self.2, 32, ptr);
            vec_xst(self.3, 48, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            _mm512_storeu_ps(ptr, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            _mm256_storeu_ps(ptr, self.0);
            _mm256_storeu_ps(ptr.add(8), self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32_x4(ptr, self.0);
        }
        #[cfg(target_arch = "loongarch64")]
        {
            lasx_xvst::<0>(transmute(self.0), transmute(ptr));
            lasx_xvst::<32>(transmute(self.1), transmute(ptr));
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            vec_xst(self.0, 0, ptr);
            vec_xst(self.1, 16, ptr);
            vec_xst(self.2, 32, ptr);
            vec_xst(self.3, 48, ptr);
        }
    }

    fn reduce_sum(&self) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            _mm512_mask_reduce_add_ps(0xFFFF, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            let mut sum = _mm256_add_ps(self.0, self.1);
            // Shift and add vector, until only 1 value left.
            // sums = [x0-x7], shift = [x4-x7]
            let mut shift = _mm256_permute2f128_ps(sum, sum, 1);
            // [x0+x4, x1+x5, ..]
            sum = _mm256_add_ps(sum, shift);
            shift = _mm256_permute_ps(sum, 14);
            sum = _mm256_add_ps(sum, shift);
            sum = _mm256_hadd_ps(sum, sum);
            let mut results: [f32; 8] = [0f32; 8];
            _mm256_storeu_ps(results.as_mut_ptr(), sum);
            results[0]
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut sum1 = vaddq_f32(self.0.0, self.0.1);
            let sum2 = vaddq_f32(self.0.2, self.0.3);
            sum1 = vaddq_f32(sum1, sum2);
            vaddvq_f32(sum1)
        }
        #[cfg(target_arch = "loongarch64")]
        {
            self.as_array().iter().sum()
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        {
            self.as_array().iter().sum()
        }
    }

    #[inline]
    fn reduce_min(&self) -> f32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            _mm512_mask_reduce_min_ps(0xFFFF, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            let mut m1 = _mm256_min_ps(self.0, self.1);
            let mut m2 = _mm256_permute2f128_ps(m1, m1, 1);
            m1 = _mm256_min_ps(m1, m2);
            m2 = _mm256_permute_ps(m1, 14);
            m1 = _mm256_min_ps(m1, m2);
            m2 = _mm256_permute_ps(m1, 1);
            m1 = _mm256_min_ps(m1, m2);
            _mm256_cvtss_f32(m1)
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let m1 = vminq_f32(self.0.0, self.0.1);
            let m2 = vminq_f32(self.0.2, self.0.3);
            let m = vminq_f32(m1, m2);
            vminvq_f32(m)
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            let m1 = lasx_xvfmin_s(self.0, self.1);
            let m2 = lasx_xvpermi_d::<14>(transmute(m1));
            let m1 = lasx_xvfmin_s(transmute(m1), transmute(m2));
            let m2 = lasx_xvpermi_w::<14>(transmute(m1), transmute(m1));
            let m1 = lasx_xvfmin_s(transmute(m1), transmute(m2));
            let m2 = lasx_xvpermi_w::<1>(transmute(m1), transmute(m1));
            let m1 = lasx_xvfmin_s(transmute(m1), transmute(m2));
            transmute(lasx_xvpickve2gr_w::<0>(transmute(m1)))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            let m1 = power10_f32_min(self.0, self.1);
            let m2 = power10_f32_min(self.2, self.3);
            let m = power10_f32_min(m1, m2);
            // 1. Shift by 8 bytes (2 words)
            let mut m_shift8: vector_float; // Declaration
            std::arch::asm!(
                "xxsldwi {0}, {1}, {1}, 2",
                out(vsreg) m_shift8,
                in(vsreg) m
            );
            let v_min = power10_f32_min(m, m_shift8);

            // 2. Shift by 4 bytes (1 word)
            let mut v_shift4: vector_float; // Declaration
            std::arch::asm!(
                "xxsldwi {0}, {1}, {1}, 1",
                out(vsreg) v_shift4,
                in(vsreg) v_min
            );
            let v_min_final = power10_f32_min(v_min, v_shift4);

            // 3. Extract and return
            let res: [f32; 4] = std::mem::transmute(v_min_final);
            res[0]
        }
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_min_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_min_ps(self.0, rhs.0), _mm256_min_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vminq_f32(self.0.0, rhs.0.0),
                vminq_f32(self.0.1, rhs.0.1),
                vminq_f32(self.0.2, rhs.0.2),
                vminq_f32(self.0.3, rhs.0.3),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfmin_s(self.0, rhs.0), lasx_xvfmin_s(self.1, rhs.1))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                power10_f32_min(self.0, rhs.0),
                power10_f32_min(self.1, rhs.1),
                power10_f32_min(self.2, rhs.2),
                power10_f32_min(self.3, rhs.3),
            )
        }
    }

    fn find(&self, val: f32) -> Option<i32> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            // let tgt = _mm512_set1_ps(val);
            // let mask = _mm512_cmpeq_ps_mask(self.0, tgt);
            // if mask != 0 {
            //     return Some(mask.trailing_zeros() as i32);
            // }
            todo!()
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            // _mm256_cmpeq_ps_mask requires "avx512l".
            for i in 0..16 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
            None
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let tgt = vdupq_n_f32(val);
            let mut arr = [0; 16];
            let mask1 = vceqq_f32(self.0.0, tgt);
            let mask2 = vceqq_f32(self.0.1, tgt);
            let mask3 = vceqq_f32(self.0.2, tgt);
            let mask4 = vceqq_f32(self.0.3, tgt);

            vst1q_u32(arr.as_mut_ptr(), mask1);
            vst1q_u32(arr.as_mut_ptr().add(4), mask2);
            vst1q_u32(arr.as_mut_ptr().add(8), mask3);
            vst1q_u32(arr.as_mut_ptr().add(12), mask4);

            for i in 0..16 {
                if arr.get_unchecked(i) != &0 {
                    return Some(i as i32);
                }
            }
            None
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            for i in 0..16 {
                if self.as_array().get_unchecked(i) == &val {
                    return Some(i as i32);
                }
            }
            None
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            let tgt = vec_splats(val);
            let masks = [
                powerpc_f32_cmpeq(self.0, tgt),
                powerpc_f32_cmpeq(self.1, tgt),
                powerpc_f32_cmpeq(self.2, tgt),
                powerpc_f32_cmpeq(self.3, tgt),
            ];

            for (i, mask) in masks.iter().enumerate() {
                if vec_any_eq(*mask, vec_splats(0xFFFFFFFFu32 as u32)) {
                    // Fallback to array check for the specific index within this vector
                    let arr: [f32; 4] = transmute(*mask);
                    for (j, &m_val) in arr.iter().enumerate() {
                        if m_val.to_bits() != 0 {
                            return Some((i * 4 + j) as i32);
                        }
                    }
                }
            }
            None
        }
    }
}

impl FloatSimd<f32, 16> for f32x16 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            self.0 = _mm512_fmadd_ps(a.0, b.0, self.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            self.0 = _mm256_fmadd_ps(a.0, b.0, self.0);
            self.1 = _mm256_fmadd_ps(a.1, b.1, self.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vfmaq_f32(self.0.0, a.0.0, b.0.0);
            self.0.1 = vfmaq_f32(self.0.1, a.0.1, b.0.1);
            self.0.2 = vfmaq_f32(self.0.2, a.0.2, b.0.2);
            self.0.3 = vfmaq_f32(self.0.3, a.0.3, b.0.3);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfmadd_s(a.0, b.0, self.0);
            self.1 = lasx_xvfmadd_s(a.1, b.1, self.1);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_madd(a.0, b.0, self.0);
            self.1 = vec_madd(a.1, b.1, self.1);
            self.2 = vec_madd(a.2, b.2, self.2);
            self.3 = vec_madd(a.3, b.3, self.3);
        }
    }
}

impl Add for f32x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_add_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_add_ps(self.0, rhs.0), _mm256_add_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vaddq_f32(self.0.0, rhs.0.0),
                vaddq_f32(self.0.1, rhs.0.1),
                vaddq_f32(self.0.2, rhs.0.2),
                vaddq_f32(self.0.3, rhs.0.3),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfadd_s(self.0, rhs.0), lasx_xvfadd_s(self.1, rhs.1))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_add(self.0, rhs.0),
                vec_add(self.1, rhs.1),
                vec_add(self.2, rhs.2),
                vec_add(self.3, rhs.3),
            )
        }
    }
}

impl AddAssign for f32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            self.0 = _mm512_add_ps(self.0, rhs.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            self.0 = _mm256_add_ps(self.0, rhs.0);
            self.1 = _mm256_add_ps(self.1, rhs.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vaddq_f32(self.0.0, rhs.0.0);
            self.0.1 = vaddq_f32(self.0.1, rhs.0.1);
            self.0.2 = vaddq_f32(self.0.2, rhs.0.2);
            self.0.3 = vaddq_f32(self.0.3, rhs.0.3);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfadd_s(self.0, rhs.0);
            self.1 = lasx_xvfadd_s(self.1, rhs.1);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_add(self.0, rhs.0);
            self.1 = vec_add(self.1, rhs.1);
            self.2 = vec_add(self.2, rhs.2);
            self.3 = vec_add(self.3, rhs.3);
        }
    }
}

impl Mul for f32x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_mul_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_mul_ps(self.0, rhs.0), _mm256_mul_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vmulq_f32(self.0.0, rhs.0.0),
                vmulq_f32(self.0.1, rhs.0.1),
                vmulq_f32(self.0.2, rhs.0.2),
                vmulq_f32(self.0.3, rhs.0.3),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfmul_s(self.0, rhs.0), lasx_xvfmul_s(self.1, rhs.1))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_mul(self.0, rhs.0),
                vec_mul(self.1, rhs.1),
                vec_mul(self.2, rhs.2),
                vec_mul(self.3, rhs.3),
            )
        }
    }
}

impl Sub for f32x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Self(_mm512_sub_ps(self.0, rhs.0))
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            Self(_mm256_sub_ps(self.0, rhs.0), _mm256_sub_ps(self.1, rhs.1))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            Self(float32x4x4_t(
                vsubq_f32(self.0.0, rhs.0.0),
                vsubq_f32(self.0.1, rhs.0.1),
                vsubq_f32(self.0.2, rhs.0.2),
                vsubq_f32(self.0.3, rhs.0.3),
            ))
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            Self(lasx_xvfsub_s(self.0, rhs.0), lasx_xvfsub_s(self.1, rhs.1))
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            Self(
                vec_sub(self.0, rhs.0),
                vec_sub(self.1, rhs.1),
                vec_sub(self.2, rhs.2),
                vec_sub(self.3, rhs.3),
            )
        }
    }
}

impl SubAssign for f32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            self.0 = _mm512_sub_ps(self.0, rhs.0)
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        unsafe {
            self.0 = _mm256_sub_ps(self.0, rhs.0);
            self.1 = _mm256_sub_ps(self.1, rhs.1);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0.0 = vsubq_f32(self.0.0, rhs.0.0);
            self.0.1 = vsubq_f32(self.0.1, rhs.0.1);
            self.0.2 = vsubq_f32(self.0.2, rhs.0.2);
            self.0.3 = vsubq_f32(self.0.3, rhs.0.3);
        }
        #[cfg(target_arch = "loongarch64")]
        unsafe {
            self.0 = lasx_xvfsub_s(self.0, rhs.0);
            self.1 = lasx_xvfsub_s(self.1, rhs.1);
        }
        #[cfg(any(target_arch = "powerpc64", target_arch = "powerpc"))]
        unsafe {
            self.0 = vec_sub(self.0, rhs.0);
            self.1 = vec_sub(self.1, rhs.1);
            self.2 = vec_sub(self.2, rhs.2);
            self.3 = vec_sub(self.3, rhs.3);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = (0..8).map(|f| f as f32).collect::<Vec<_>>();
        let b = (10..18).map(|f| f as f32).collect::<Vec<_>>();

        let mut simd_a = unsafe { f32x8::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { f32x8::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        assert!(
            (0..8)
                .zip(simd_add.as_array().iter())
                .all(|(x, &y)| (x + x + 10) as f32 == y)
        );

        let simd_mul = simd_a * simd_b;
        assert!(
            (0..8)
                .zip(simd_mul.as_array().iter())
                .all(|(x, &y)| (x * (x + 10)) as f32 == y)
        );

        let simd_sub = simd_b - simd_a;
        assert!(simd_sub.as_array().iter().all(|&v| v == 10.0));

        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -80.0);

        let mut simd_power = f32x8::splat(0.0);
        simd_power.multiply_add(simd_a, simd_a);

        assert_eq!(
            "f32x8([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])",
            format!("{:?}", simd_power)
        );
    }

    #[test]
    fn test_f32x8_cmp_ops() {
        let a = [1.0_f32, 2.0, 5.0, 6.0, 7.0, 3.0, 2.0, 1.0];
        let b = [2.0_f32, 1.0, 4.0, 5.0, 9.0, 5.0, 6.0, 2.0];
        let c = [2.0_f32, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0];
        let simd_a: f32x8 = (&a).into();
        let simd_b: f32x8 = (&b).into();
        let simd_c: f32x8 = (&c).into();

        let min_simd = simd_a.min(&simd_b);
        assert_eq!(
            min_simd.as_array(),
            [1.0, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0]
        );
        let min_val = min_simd.reduce_min();
        assert_eq!(min_val, 1.0);
        let min_val = simd_c.reduce_min();
        assert_eq!(min_val, 1.0);

        assert_eq!(Some(2), simd_a.find(5.0));
        assert_eq!(Some(1), simd_a.find(2.0));
        assert_eq!(None, simd_a.find(-200.0));
    }

    #[test]
    fn test_basic_f32x16_ops() {
        let a = (0..16).map(|f| f as f32).collect::<Vec<_>>();
        let b = (10..26).map(|f| f as f32).collect::<Vec<_>>();

        let mut simd_a = unsafe { f32x16::load_unaligned(a.as_ptr()) };
        let simd_b = unsafe { f32x16::load_unaligned(b.as_ptr()) };

        let simd_add = simd_a + simd_b;
        assert!(
            (0..16)
                .zip(simd_add.as_array().iter())
                .all(|(x, &y)| (x + x + 10) as f32 == y)
        );

        let simd_mul = simd_a * simd_b;
        assert!(
            (0..16)
                .zip(simd_mul.as_array().iter())
                .all(|(x, &y)| (x * (x + 10)) as f32 == y)
        );

        simd_a -= simd_b;
        assert_eq!(simd_a.reduce_sum(), -160.0);

        let mut simd_power = f32x16::zeros();
        simd_power.multiply_add(simd_a, simd_a);

        assert_eq!(
            format!("f32x16({:?})", [100.0; 16]),
            format!("{:?}", simd_power)
        );
    }

    #[test]
    fn test_f32x16_cmp_ops() {
        let a = [
            1.0_f32, 2.0, 5.0, 6.0, 7.0, 3.0, 2.0, 1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0,
        ];
        let b = [
            2.0_f32, 1.0, 4.0, 5.0, 9.0, 5.0, 6.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 1.0,
        ];
        let c = [
            1.0_f32, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, -1.0,
        ];
        let simd_a: f32x16 = (&a).into();
        let simd_b: f32x16 = (&b).into();
        let simd_c: f32x16 = (&c).into();

        let min_simd = simd_a.min(&simd_b);
        assert_eq!(
            min_simd.as_array(),
            [
                1.0, 1.0, 4.0, 5.0, 7.0, 3.0, 2.0, 1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0
            ]
        );
        let min_val = min_simd.reduce_min();
        assert_eq!(min_val, -0.5);
        let min_val = simd_c.reduce_min();
        assert_eq!(min_val, -1.0);

        assert_eq!(Some(2), simd_a.find(5.0));
        assert_eq!(Some(1), simd_a.find(2.0));
        assert_eq!(Some(13), simd_a.find(9.0));
        assert_eq!(None, simd_a.find(-200.0));
    }

    #[test]
    fn test_f32x8_gather() {
        let a = (0..256).map(|f| f as f32).collect::<Vec<_>>();
        let idx = [0_i32, 4, 8, 12, 16, 20, 24, 29];
        let v = f32x8::gather(&a, &idx);
        assert_eq!(v.reduce_sum(), 113.0);
    }
}
