package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Initializer;
import com.maaza.xyz.xxx.Param;

public final class Dense implements Layer {

	private final int in;
	private final int out;
	private final Param bias;
	private final Param weight;
	private final Param[] params;
	private Array cache;

	public Dense(final int in, final int out, final Initializer biasInit, final Initializer weightInit) {
		this.in = in;
		this.out = out;
		this.bias = new Param(this.out);
		this.weight = new Param(this.out, this.in);
		this.params = new Param[] { this.bias, this.weight };
		biasInit.init(this.bias.data);
		weightInit.init(this.weight.data);
	}

	@Override
	public final void forward(final Array input, final Array output) {
		this.cache = input;
		final float[] x = input.data;
		final float[] y = output.data;
		final float[] b = this.bias.data.data;
		final float[] w = this.weight.data.data;
		final int in = this.in;
		final int out = this.out;
		final int size = x.length / in;
		for (int k = 0; k < size; k++) {
			final int xoff = k * in;
			final int yoff = k * out;
			for (int j = 0, woff = 0; j < out; j++, woff += in) {
				float data = b[j];
				for (int i = 0; i < in; i++) {
					data += x[xoff + i] * w[woff + i];
				}
				y[yoff + j] = data;
			}
		}
	}

	@Override
	public final void backward(final Array output, final Array input) {
		final float[] x = this.cache.data;
		final float[] dx = input.data;
		final float[] dy = output.data;
		final float[] db = this.bias.grad.data;
		final float[] dw = this.weight.grad.data;
		final float[] w = this.weight.data.data;
		final int in = this.in;
		final int out = this.out;
		final int size = x.length / in;
		for (int k = 0; k < size; k++) {
			final int xoff = k * in;
			final int yoff = k * out;
			for (int j = 0, woff = 0; j < out; j++, woff += in) {
				final float grad = dy[yoff + j];
				db[j] += grad;
				for (int i = 0; i < in; i++) {
					dw[woff + i] += x[xoff + i] * grad;
					dx[xoff + i] += w[woff + i] * grad;
				}
			}
		}
	}

	@Override
	public final int[] shape(final int... size) {
		final int[] shape = size.clone();
		shape[shape.length - 1] = this.out;
		return shape;
	}

	@Override
	public final Param[] params() {
		return this.params;
	}
}
