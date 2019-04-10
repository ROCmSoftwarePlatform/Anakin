/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
 
#pragma once

#include <string>
#include <vector>

namespace TensileConv
{
	enum class E_TCRelu
	{
		NORELU = 0,
		RELU = 1,
		PRELU = 2
	};
	enum class E_TCSearch
	{
		NOSEARCH = 0,	// 不search.   若db有解,则返回最优解; 否则返回时间为-1秒
		AUTO = 1,		// 自动search. 若db有解,则返回最优解; 否则执行自动搜索后返回最优解
		BRUTE = 2,		// 强制暴力搜索寻找最优解
		GENETIC = 3		// 强制使用遗传算法搜索最优解
	};

	typedef struct TCSolutionType
	{
		std::string kernel_file;
		std::string kernel_name;
		std::vector<int> ParamSize;
		std::vector<int> GroupSize;
		std::vector<int> GlobalSize;
	} T_TCSolution;

	// direct convolution 1x1 forward
	class DirConv1x1Fwd
	{
	public:
		DirConv1x1Fwd();
		~DirConv1x1Fwd();

		static void SetDbFilePath(std::string path);
		static std::string GetDbFilePath();
		double TuneProblem(int W, int H, int C, int K, int N, int U, int V,
			bool bias, E_TCRelu relu, E_TCSearch search,
			T_TCSolution & solution);

	private:
		std::string kernelFile;
		std::string kernelName;
		std::vector<int> paramSize;
		std::vector<int> groupSize;
		std::vector<int> globalSize;
		double timeSec;
	};
}

