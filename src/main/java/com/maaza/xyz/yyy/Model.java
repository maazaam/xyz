package com.maaza.xyz.yyy;

import java.util.Arrays;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public final class Model {

	private final Layer[] layer;
	private final Param[] param;
	private final Array[] data;
	private final Array[] grad;

	public Model(final int[] size, final Layer... layer) {
		this.layer = layer.clone();
		int num = 0;
		for (final Layer item : this.layer) {
			num += item.params().length;
		}
		this.param = new Param[num];
		int off = 0;
		for (final Layer item : this.layer) {
			final Param[] src = item.params();
			final int len = src.length;
			System.arraycopy(src, 0, this.param, off, len);
			off += len;
		}
		final int len = this.layer.length - 1;
		this.data = new Array[len];
		this.grad = new Array[len];
		int[] shape = size.clone();
		for (int i = 0; i < len; i++) {
			shape = this.layer[i].shape(shape);
			this.data[i] = new Array(shape);
			this.grad[i] = new Array(shape);
		}
	}

	public final void forward(final Array input, final Array output) {
		Array x = input;
		final Layer[] layer = this.layer;
		final Array[] data = this.data;
		final int len = layer.length;
		for (int i = 0; i < len; i++) {
			final Array y = i == len - 1 ? output : data[i];
			layer[i].forward(x, y);
			x = y;
		}
	}

	public final void backward(final Array output, final Array input) {
		Array dy = output;
		final Layer[] layer = this.layer;
		final Array[] grad = this.grad;
		final int len = layer.length;
		for (int i = len - 1; i >= 0; i--) {
			final Array dx = i == 0 ? input : grad[i - 1];
			layer[i].backward(dy, dx);
			dy = dx;
		}
	}

	public final void clear() {
		for (final Param item : this.param) {
			Arrays.fill(item.grad.data, 0.0f);
		}
		for (final Array item : this.grad) {
			Arrays.fill(item.data, 0.0f);
		}
	}

	public final Param[] params() {
		return this.param;
	}
}
