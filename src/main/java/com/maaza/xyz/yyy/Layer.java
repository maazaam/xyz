package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public interface Layer {

	public void forward(Array input, Array output);

	public void backward(Array output, Array input);

	public int[] shape(int... size);

	public Param[] params();
}
