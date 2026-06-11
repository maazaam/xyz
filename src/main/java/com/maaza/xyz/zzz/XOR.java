package com.maaza.xyz.zzz;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Calculator;
import com.maaza.xyz.xxx.Calculators;
import com.maaza.xyz.xxx.Initializers;
import com.maaza.xyz.xxx.Optimizer;
import com.maaza.xyz.xxx.Optimizers;
import com.maaza.xyz.yyy.Dense;
import com.maaza.xyz.yyy.Model;
import com.maaza.xyz.yyy.TANH;

public final class XOR {

	public static final void main(final String[] args) {
		// XOR inputs
		//
		// [0,0] -> 0
		// [0,1] -> 1
		// [1,0] -> 1
		// [1,1] -> 0

		final Array input = new Array(4, 2);

		input.data[0] = 0.0f;
		input.data[1] = 0.0f;

		input.data[2] = 0.0f;
		input.data[3] = 1.0f;

		input.data[4] = 1.0f;
		input.data[5] = 0.0f;

		input.data[6] = 1.0f;
		input.data[7] = 1.0f;

		// XOR targets

		final Array target = new Array(4, 1);

		target.data[0] = 0.0f;
		target.data[1] = 1.0f;
		target.data[2] = 1.0f;
		target.data[3] = 0.0f;

		// model

		final Model model = new Model(

				new int[] { 4, 2 },

				new Dense(2, 8, Initializers.fill(0.0f), Initializers.xavi()),

				new TANH(),

				new Dense(8, 1, Initializers.fill(0.0f), Initializers.xavi()));

		// optimizer

		final Optimizer opt = Optimizers.adam(0.01f, 0.9f, 0.999f);

		// loss

		final Calculator loss = Calculators.bin();

		// output tensor

		final Array output = new Array(4, 1);

		// output gradient tensor

		final Array grad = new Array(4, 1);

		// input gradient tensor

		final Array dx = new Array(4, 2);

		// train

		final int epochs = 10000;

		for (int epoch = 0; epoch < epochs; epoch++) {

			// clear previous gradients

			model.clear();

			// forward

			model.forward(input, output);

			// compute loss + output gradients

			final float val = loss.loss(target, output, grad);

			// backward

			model.backward(grad, dx);

			// optimizer step

			opt.step(model.params());

			// print progress

			if (epoch % 1000 == 0) {

				System.out.printf("epoch=%d loss=%.6f%n", epoch, val);
			}
		}

		// final predictions

		System.out.println();
		System.out.println("predictions:");

		model.forward(input, output);

		for (int i = 0; i < 4; i++) {

			final float x0 = input.data[i * 2];

			final float x1 = input.data[i * 2 + 1];

			final float y = output.data[i];

			final float p = 1.0f / (1.0f + (float) Math.exp(-y));

			System.out.printf("%.0f xor %.0f -> %.6f%n", x0, x1, p);
		}
	}
}
