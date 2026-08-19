package com.maaza.xyz;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Utils {

	private static final RandomGeneratorFactory<RandomGenerator> FACT = RandomGeneratorFactory.of("L64X128MixRandom");
	private static final float EPS1 = 1.0e-7f;
	private static final float EPS2 = 1.0e-8f;

	private Utils() {
	}

	public static final RandomGenerator rng() {
		return FACT.create();
	}

	public static final RandomGenerator rng(final long sed) {
		return FACT.create(sed);
	}

	public static final void init_fill(final float[] arr, final float val) {
		final int len = arr.length;
		Arrays.fill(arr, 0, len, val);
	}

	public static final void init_copy(final float[] arr, final float[] src) {
		final int len = arr.length;
		System.arraycopy(src, 0, arr, 0, len);
	}

	public static final void init_rndu(final float[] arr, final float min, final float max, final RandomGenerator rng) {
		final int len = arr.length;
		final float val = max - min;
		for (int i = 0; i < len; i++) {
			arr[i] = rng.nextFloat() * val + min;
		}
	}

	public static final void init_rndn(final float[] arr, final float avg, final float sdv, final RandomGenerator rng) {
		final int len = arr.length;
		for (int i = 0; i < len; i++) {
			arr[i] = (float) (rng.nextGaussian() * sdv + avg);
		}
	}

	public static final float xavi_rndu(final int row, final int col) {
		return (float) Math.sqrt(6.0 / (row + col));
	}

	public static final float xavi_rndn(final int row, final int col) {
		return (float) Math.sqrt(2.0 / (row + col));
	}

	public static final float kaim_rndu(final int row) {
		return (float) Math.sqrt(6.0 / row);
	}

	public static final float kaim_rndn(final int row) {
		return (float) Math.sqrt(2.0 / row);
	}

	public static final float loss_mse(final float[] real, final float[] pred, final float[] grad) {
		final int len = real.length;
		final float mul = 1.0f / len;
		float out = 0.0f;
		for (int i = 0; i < len; i++) {
			final float dif = pred[i] - real[i];
			out += dif * dif;
			grad[i] = dif * mul;
		}
		return 0.5f * out * mul;
	}

	public static final float loss_bce(final float[] real, final float[] pred, final float[] grad) {
		final int len = real.length;
		final float mul = 1.0f / len;
		float out = 0.0f;
		for (int i = 0; i < len; i++) {
			final float ri = real[i];
			final float val = Math.max(EPS1, Math.min(1.0f - EPS1, pred[i]));
			final float lgv = (float) Math.log(val);
			final float lgr = (float) Math.log1p(-val);
			out += ri * lgv + (1.0f - ri) * lgr;
			grad[i] = (val - ri) / (val * (1.0f - val)) * mul;
		}
		return -out * mul;
	}

	public static final float loss_bin(final float[] real, final float[] pred, final float[] grad) {
		final int len = real.length;
		final float mul = 1.0f / len;
		float out = 0.0f;
		for (int i = 0; i < len; i++) {
			final float ri = real[i];
			final float pi = pred[i];
			final float max = Math.max(0.0f, pi);
			final float exp = (float) Math.exp(-Math.abs(pi));
			final float log = (float) Math.log1p(exp);
			out += max - pi * ri + log;
			final float inv = 1.0f / (1.0f + exp);
			final float sig = pi >= 0.0f ? inv : exp * inv;
			grad[i] = (sig - ri) * mul;
		}
		return out * mul;
	}

	public static final float loss_mul(final float[] real, final float[] pred, final float[] grad, final int batch, final int clazz) {
		final float mul = 1.0f / batch;
		float out = 0.0f;
		for (int j = 0; j < batch; j++) {
			final int off = j * clazz;
			final int end = off + clazz;
			float max = pred[off];
			for (int i = off + 1; i < end; i++) {
				max = Math.max(max, pred[i]);
			}
			float sum = 0.0f;
			for (int i = off; i < end; i++) {
				final float exp = (float) Math.exp(pred[i] - max);
				grad[i] = exp;
				sum += exp;
			}
			final float log = (float) Math.log(sum);
			final float inv = 1.0f / sum;
			for (int i = off; i < end; i++) {
				final float ri = real[i];
				final float smx = grad[i] * inv;
				if (ri > 0.0f) {
					out += ri * (log - (pred[i] - max));
				}
				grad[i] = (smx - ri) * mul;
			}
		}
		return out * mul;
	}

	public static final void opti_mntm(final float[] data, final float[] grad, final float[] v, final float rate, final float beta) {
		final int len = data.length;
		final float bm = 1.0f - beta;
		for (int i = 0; i < len; i++) {
			final float vi = v[i] = beta * v[i] + bm * grad[i];
			data[i] -= rate * vi;
		}
	}

	public static final void opti_adam(final float[] data, final float[] grad, final float[] m, final float[] v, final float alpha, final float beta1, final float beta2) {
		final int len = data.length;
		final float b1m = 1.0f - beta1;
		final float b2m = 1.0f - beta2;
		for (int i = 0; i < len; i++) {
			final float gi = grad[i];
			final float mi = m[i] = beta1 * m[i] + b1m * gi;
			final float vi = v[i] = beta2 * v[i] + b2m * gi * gi;
			data[i] -= alpha * mi / ((float) Math.sqrt(vi) + EPS2);
		}
	}

	public static final float alpha(final float rate, final float beta1, final float beta2, final float[] bt) {
		final float b1t = bt[0] *= beta1;
		final float b2t = bt[1] *= beta2;
		return rate * (float) Math.sqrt(1.0f - b2t) / (1.0f - b1t);
	}

	public static final float[] bt() {
		return new float[] { 1.0f, 1.0f };
	}
}
