package com.maaza.xyz.xxx;

public final class Param {

	public final Array data;
	public final Array grad;
	public Object state;

	public Param(final int... size) {
		this.data = new Array(size);
		this.grad = new Array(size);
	}
}
