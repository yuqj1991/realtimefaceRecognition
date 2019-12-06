
#ifndef PCA_DEMENSION_HPP_
#define PCA_DEMENSION_HPP_


#include <iostream>
#include <algorithm>
#include <fstream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

 
MatrixXd featurnormail(MatrixXd &X)
{
	//计算每一维的均值
	MatrixXd X1 = X.transpose();
	MatrixXd meanval = X1.colwise().mean();
 
//	cout << "meanval" << endl << meanval <<endl;
	//样本均值化为0
	RowVectorXd meanvecRow = meanval;
	X1.rowwise() -= meanvecRow;
	return X1.transpose();
}
void ComComputeCov(MatrixXd &X, MatrixXd &C)
{
	//计算协方差矩阵
	C = X*X.adjoint();//相当于XT*X adjiont()求伴随矩阵 相当于求矩阵的转置
	C = C.array() / X.cols();//C.array()矩阵的数组形式
}
 
void ComputEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
{
	//计算特征向量和特征值 SelfAdjointEigenSolver自动将计算得到的特征向量和特征值排序
	SelfAdjointEigenSolver<MatrixXd> eig(C);
	vec = eig.eigenvectors();
	val = eig.eigenvalues();
}
 
// 计算维度
int ComputDim(MatrixXd &val)
{
	int dim;
	double sum = 0;
	for (int i = val.rows() - 1; i >= 0;--i)
	{
		sum += val(i, 0);
		dim = i;
		if (sum / val.sum()>=0.8)//达到所需要的要求时
			break;
	}
	return val.rows() - dim;
}

#endif
