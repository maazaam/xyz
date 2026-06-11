package com.maaza.xyz.xxx;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Initializers {

	private static final RandomGeneratorFactory<RandomGenerator> FACT = RandomGeneratorFactory.of("L64X128MixRandom");

	private Initializers() {
	}

	private static final int[] fans(final int[] arr) {
		final int len = arr.length;
		int val = 1;
		for (int i = 2; i < len; i++) {
			val *= arr[i];
		}
		return new int[] { arr[1] * val, arr[0] * val };
	}

	public static final Initializer fill(final float val) {
		return (x) -> {
			final float[] arr = x.data;
			Arrays.fill(arr, val);
		};
	}

	public static final Initializer copy(final float[] src) {
		return (x) -> {
			final float[] arr = x.data;
			final int len = arr.length;
			System.arraycopy(src, 0, arr, 0, len);
		};
	}

	public static final Initializer rndu(final float min, final float max) {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			final float[] arr = x.data;
			final int len = arr.length;
			for (int i = 0; i < len; i++) {
				arr[i] = (float) rng.nextDouble(min, max);
			}
		};
	}

	public static final Initializer rndn(final float avg, final float sdv) {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			final float[] arr = x.data;
			final int len = arr.length;
			for (int i = 0; i < len; i++) {
				arr[i] = (float) rng.nextGaussian(avg, sdv);
			}
		};
	}

	public static final Initializer xavi() {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			final int[] fan = fans(x.size);
			final float val = (float) Math.sqrt(6.0f / (fan[0] + fan[1]));
			final float[] arr = x.data;
			final int len = arr.length;
			for (int i = 0; i < len; i++) {
				arr[i] = (float) rng.nextDouble(-val, val);
			}
		};
	}

	public static final Initializer kaim() {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			final int[] fan = fans(x.size);
			final float val = (float) Math.sqrt(2.0f / fan[0]);
			final float[] arr = x.data;
			final int len = arr.length;
			for (int i = 0; i < len; i++) {
				arr[i] = (float) rng.nextGaussian(0.0f, val);
			}
		};
	}
}
