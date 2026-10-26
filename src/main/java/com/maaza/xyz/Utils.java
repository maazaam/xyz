package com.maaza.xyz;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Utils {

	private static final RandomGeneratorFactory<RandomGenerator> FACT = RandomGeneratorFactory.of("L64X128MixRandom");

	private Utils() {
	}

	public static final RandomGenerator rng() {
		return FACT.create();
	}

	public static final RandomGenerator rng(final long sed) {
		return FACT.create(sed);
	}

	public static final int[] fans(final int[] shp) {
		int val = 1;
		for (int i = 2; i < shp.length; i++) {
			val *= shp[i];
		}
		return new int[] { shp[1] * val, shp[0] * val };
	}

	public static final void init_fill(final float[] arr, final float val) {
		Arrays.fill(arr, val);
	}

	public static final void init_copy(final float[] arr, final float[] src) {
		System.arraycopy(src, 0, arr, 0, arr.length);
	}

	public static final void init_rndu(final float[] arr, final float min, final float max, final RandomGenerator rng) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = rng.nextFloat(min, max);
		}
	}

	public static final void init_rndn(final float[] arr, final float avg, final float sdv, final RandomGenerator rng) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = (float) rng.nextGaussian(avg, sdv);
		}
	}

	public static final void init_xavi(final float[] arr, final int row, final int col, final RandomGenerator rng) {
		final float val = (float) Math.sqrt(6.0f / (row + col));
		init_rndu(arr, -val, val, rng);
	}

	public static final void init_kaim(final float[] arr, final int row, final RandomGenerator rng) {
		final float val = (float) Math.sqrt(2.0f / row);
		init_rndn(arr, 0.0f, val, rng);
	}

	public static final float acti_relu(final float val) {
		return Math.max(0.0f, val);
	}

	public static final float acti_sigm(final float val) {
		if (val >= 0.0f) {
			final float exp = (float) Math.exp(-val);
			return 1.0f / (1.0f + exp);
		} else {
			final float exp = (float) Math.exp(val);
			return exp / (1.0f + exp);
		}
	}

	public static final float acti_tanh(final float val) {
		return (float) Math.tanh(val);
	}

	public static final float deri_relu(final float val) {
		return val > 0.0f ? 1.0f : 0.0f;
	}

	public static final float deri_sigm(final float val) {
		return val * (1.0f - val);
	}

	public static final float deri_tanh(final float val) {
		return 1.0f - val * val;
	}
}
