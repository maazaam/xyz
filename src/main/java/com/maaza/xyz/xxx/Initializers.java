package com.maaza.xyz.xxx;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Initializers {

	private static final RandomGeneratorFactory<RandomGenerator> FACT = RandomGeneratorFactory.of("L64X128MixRandom");

	private Initializers() {
	}

	public static final Initializer fill(final float val) {
		return (x) -> {
			Arrays.fill(x, val);
		};
	}

	public static final Initializer copy(final float[] arr) {
		return (x) -> {
			System.arraycopy(arr, 0, x, 0, x.length);
		};
	}

	public static final Initializer rndu(final float min, final float max) {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			for (int i = 0; i < x.length; i++) {
				x[i] = (float) rng.nextDouble(min, max);
			}
		};
	}

	public static final Initializer rndn(final float avg, final float sdv) {
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			for (int i = 0; i < x.length; i++) {
				x[i] = (float) rng.nextGaussian(avg, sdv);
			}
		};
	}

	public static final Initializer xavi(final int in, final int out) {
		final float val = (float) Math.sqrt(6.0f / (in + out));
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			for (int i = 0; i < x.length; i++) {
				x[i] = (float) rng.nextDouble(-val, val);
			}
		};
	}

	public static final Initializer kaim(final int in) {
		final float val = (float) Math.sqrt(2.0f / in);
		final RandomGenerator rng = FACT.create();
		return (x) -> {
			for (int i = 0; i < x.length; i++) {
				x[i] = (float) rng.nextGaussian(0.0f, val);
			}
		};
	}
}
