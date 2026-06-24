package com.maaza.xyz.zzz;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Calculator;
import com.maaza.xyz.xxx.Calculators;
import com.maaza.xyz.xxx.Initializer;
import com.maaza.xyz.xxx.Initializers;
import com.maaza.xyz.xxx.Optimizer;
import com.maaza.xyz.xxx.Optimizers;
import com.maaza.xyz.xxx.Param;
import com.maaza.xyz.yyy.Dense;
import com.maaza.xyz.yyy.Model;
import com.maaza.xyz.yyy.TANH;

public final class XOR {

	private static final void benchmark(final Runnable task, final String name, final int warmup, final int measure) {

		System.out.printf("%n========== Benchmark: %s ==========%n", name);
		System.out.printf("Warmup: %d, Measure: %d%n%n", warmup, measure);

		for (int i = 0; i < warmup; i++) {
			task.run();
		}

		final Runtime runtime = Runtime.getRuntime();

		long totalTime = 0;
		long totalMemory = 0;

		for (int i = 0; i < measure; i++) {

			System.gc();

			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}

			final long before = runtime.totalMemory() - runtime.freeMemory();
			final long start = System.nanoTime();

			task.run();

			final long stop = System.nanoTime();
			final long after = runtime.totalMemory() - runtime.freeMemory();

			final long time = stop - start;
			final long memory = after - before;

			System.out.printf("Run %2d | Time: %.3f ms | Memory: %.2f kb%n", i + 1, time / 1_000_000.0, memory / 1_024.0);

			totalTime += time;
			totalMemory += memory;

		}

		final double averageTime = totalTime / (measure * 1_000_000.0);
		final double averageMemory = totalMemory / (measure * 1_024.0);

		System.out.printf("%nAverage Time   : %.3f ms", averageTime);
		System.out.printf("%nAverage Memory : %.2f kb%n", averageMemory);

	}

	private static final void task() {

		final int bs = 4;

		final int in = 2;
		final int hn = 4;
		final int on = 1;

		final Array x = new Array(bs, in);
		x.data[0] = 0.0f;
		x.data[1] = 0.0f;
		x.data[2] = 0.0f;
		x.data[3] = 1.0f;
		x.data[4] = 1.0f;
		x.data[5] = 0.0f;
		x.data[6] = 1.0f;
		x.data[7] = 1.0f;

		final Array z = new Array(bs, on);
		z.data[0] = 0.0f;
		z.data[1] = 1.0f;
		z.data[2] = 1.0f;
		z.data[3] = 0.0f;

		final Initializer fill = Initializers.fill(0.0f);
		final Initializer xavi = Initializers.xavi();

		final Model model = new Model(

				new int[] { bs, in },

				new Dense(in, hn, fill, xavi),

				new TANH(),

				new Dense(hn, on, fill, xavi));

		final Optimizer opti = Optimizers.adam(0.01f, 0.9f, 0.999f);

		final Calculator calc = Calculators.bin();

		final Array y = new Array(bs, on);
		final Array dy = new Array(bs, on);
		final Array dx = new Array(bs, in);

		final Param[] pms = model.params();

		final int epochs = 10000;

		for (int epoch = 0; epoch < epochs; epoch++) {

			model.clear();

			model.forward(x, y);

			final float loss = calc.loss(z, y, dy);

			model.backward(dy, dx);

			opti.step(pms);

			if (epoch % 500 == 0) {
				System.out.printf("epoch=%d loss=%.6f%n", epoch, loss);
			}

		}

		System.out.println();
		System.out.println("Predictions:");

		model.forward(x, y);

		for (int i = 0; i < bs; i++) {

			final float x0 = x.data[i * 2];
			final float x1 = x.data[i * 2 + 1];

			final float y0 = y.data[i];
			final float y1 = 1.0f / (1.0f + (float) Math.exp(-y0));

			System.out.printf("[%.0f %.0f] -> %.6f%n", x0, x1, y1);

		}

	}

	public static final void main(final String[] args) {
		// task();
		benchmark(() -> task(), "xor", 10, 5);
	}
}
