#include "function.h"
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

cv::Mat SK_Weighted_Guided_Image_Filter_8UC1(const cv::Mat& _im, const cv::Mat& _p, int r, int r2, double eps, double lamda, int N, cv::Mat& a_avg);
void CLAHE8UC1(const cv::Mat& src, cv::Mat& dst, double clip, cv::Size tiles);

int main()
{
	cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
 
	Mat image = imread("C:/Users/24563/Desktop/infraredProject/dataset/Input Images Used/B1.png",0);
    imshow("image_gray", image);

	//获取基础层图像
	cv::Mat a_avg(image.rows, image.cols, CV_64FC1);
	Mat skwgif_base = SK_Weighted_Guided_Image_Filter_8UC1(image, image, 2, 3, 0.16, 0.001 * 0.001 * 255 * 255, 640 * 512,a_avg);
    cv::normalize(skwgif_base, skwgif_base, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //imshow("skwgif_base", skwgif_base);

    // CLAHE对比度增强
    Mat clahe_img(cv::Size(WIDTH, HEIGHT), CV_8UC1);
    CLAHE8UC1(skwgif_base, clahe_img, 4, cv::Size(2, 2));
    //imshow("clahe_image", clahe_img);

    //cv::Mat blurred_image;
    Mat skwgif_gaussian_base(cv::Size(WIDTH, HEIGHT), CV_8UC1);
    cv::GaussianBlur(skwgif_base, skwgif_gaussian_base, cv::Size(7, 7), 0, 0);


    //细节提取
    cv::Mat skwgif_gaussian_detail = skwgif_gaussian_base;
    const int length = skwgif_gaussian_detail.rows * skwgif_gaussian_detail.cols;
    for (int i = 0; i < length; ++i)
        skwgif_gaussian_detail.data[i] = cv::saturate_cast<uchar>((image.data[i] - skwgif_gaussian_detail.data[i]));
    imshow("skwgif_gaussian_detail", skwgif_gaussian_detail);

	//细节增强
	cv::Mat detail_enhancement= skwgif_gaussian_detail;
	uchar* p_detail_enhancement;
	uchar* p_skwgif_gaussian_detail;
	double* p_a_avg;
	for (int i = 0; i < image.rows; i++)
	{
		p_detail_enhancement = detail_enhancement.ptr<uchar>(i);//获取每行首地址
		p_skwgif_gaussian_detail = skwgif_gaussian_detail.ptr<uchar>(i);
		p_a_avg = a_avg.ptr<double>(i);
		for (int j = 0; j < image.cols; ++j)
		{
			p_detail_enhancement[j] = (10*p_a_avg[j]+5)* p_skwgif_gaussian_detail[j];
		}
	}
	imshow("detail_enhancement", detail_enhancement);


    //合并两层图像
    cv::Mat combine_img = clahe_img.clone();
    for (int i = 0; i < clahe_img.rows; i++) {
        for (int j = 0; j < clahe_img.cols; j++) {
            combine_img.at<uchar>(i, j) = cv::saturate_cast<uchar>(clahe_img.at<uchar>(i, j) * 0.9 + detail_enhancement.at<uchar>(i, j) * 0.1);
        }
    }
    imshow("combine_img", combine_img);

	waitKey(0);
	return 0;
}


vector<vector<array<double, 18>>> Grad_func(const cv::Mat& img) {

    // 创建一个640x512的矩阵，每个元素都是一个大小为18的数组
    vector<vector<array<double, 18>>> matrix(img.rows, vector<array<double, 18>>(img.cols));

    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            cv::Mat Gij(2, 9, CV_64FC1); // 存储局部梯度矩阵
            cv::Mat Gij_1 = cv::Mat::zeros(3, 3, CV_64FC1);
            cv::Mat Gij_2 = cv::Mat::zeros(3, 3, CV_64FC1);

            // 提取当前像素点周围的 3x3 区域作为感兴趣区域（ROI）
            //其范围是从 (i - 1, j - 1) 到 (i + 1, j + 1) 的矩形区域（包括左上角和右下角）。
            cv::Mat roi = img(cv::Range(i - 1, i + 2), cv::Range(j - 1, j + 2));        

            cv::Sobel(roi, Gij_1, CV_64F, 1, 0, 1, 1, 0, BORDER_DEFAULT);
            cv::Sobel(roi, Gij_2, CV_64F, 0, 1, 1, 1, 0, BORDER_DEFAULT);

            // 将局部梯度矩阵存储到 Grad_mat 中
            Gij_1 = Gij_1.reshape(1, 9);
            Gij_2 = Gij_2.reshape(1, 9);

            cv::hconcat(Gij_1, Gij_2, Gij);// 垂直拼接
            
            for (int k = 0; k < 9; ++k) {
                matrix[i][j][k] = (double)Gij_1.at<double>(0, k);
            }
            for (int k = 9; k < 18; ++k) {
                matrix[i][j][k] = (double)Gij_2.at<double>(0, k - 9);
            }
        }
    }

    return matrix;
}

vector<vector<array<double, 9>>> steering_kernel(const cv::Mat& img) {

    vector<vector<array<double, 18>>> Grad_mat = Grad_func(img);
    //W 是一个二维列表，大小与输入图像的行列数相同，每个元素都是一个 3x3 的矩阵。
    vector<vector<array<double, 9>>> W(img.rows, vector<array<double, 9>>(img.cols));

    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {     

            // Reshape the 2x2 sub-matrix to a 2x2 2D matrix
            cv::Mat reshaped_grad(9, 2, CV_64F); // 创建一个9x2的双精度矩阵
            int index = 0; // 用于遍历数组元素的索引
            for (int ii = 0; ii < 9; ++ii) {
                for (int jj = 0; jj < 2; ++jj) {
                    reshaped_grad.at<double>(ii, jj) = Grad_mat[i][j][index]; // 将数组中的元素逐个复制到矩阵中的对应位置
                    index++; // 更新索引
                }
            }

            // Perform SVD on the reshaped 2x2 matrix
            cv::Mat u, vt, s;
            cv::SVDecomp(reshaped_grad, s, u, vt);
            cv::Mat v = vt.t();
            cv::Vec2d v2 = v.at<cv::Vec2d>(1);

            double theta = v2[1] == 0 ? CV_PI / 2 : std::atan(v2[0] / v2[1]);
            double sigma = (s.at<double>(0) + 1.0) / (s.at<double>(1) + 1.0);
            double gamma = std::sqrt(((s.at<double>(0) * s.at<double>(1)) + 0.01) / 9);
            cv::Matx22d Rot_mat(std::cos(theta), std::sin(theta), -std::sin(theta), std::cos(theta));
            cv::Matx22d El_mat(sigma, 0, 0, 1 / sigma);
            cv::Matx22d C = gamma * (Rot_mat * El_mat * Rot_mat.t());
            double coeff = std::sqrt(cv::determinant(C)) / (2 * CV_PI * 5.76);

            int flag = 0;
            for (int n_i = i - 1; n_i <= i + 1; ++n_i) {
                for (int n_j = j - 1; n_j <= j + 1; ++n_j) {
                    cv::Vec2i xik(n_j - j, n_i - i);
                    double wik = coeff * std::exp(-cv::Matx21d(xik).dot(C * cv::Matx21d(xik)) / 11.52);
                    W[i][j][flag] = wik;
                    flag++;
                }
            }
        }
    }

    return W;
}


cv::Mat SK_Weighted_Guided_Image_Filter_8UC1(const cv::Mat& _im, const cv::Mat& _p, int r, int r2, double eps, double lamda, int N,cv::Mat& a_avg) {
    cv::Mat mean_I, mean_I2, mean_p, mean_p2, corr_I, corr_I2, corr_Ip;

    cv::Mat im, p;
    _im.convertTo(im, CV_64FC1);
    _p.convertTo(p, CV_64FC1);

    for (int i = 0; i < im.rows; ++i) {
        for (int j = 0; j < im.cols; ++j) {
            im.at<double>(i, j) = (double)im.at<double>(i, j) / 255;
            p.at<double>(i, j) = (double)p.at<double>(i, j) / 255;

        }
    }

    cv::boxFilter(im, mean_I, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(im, mean_I2, CV_64FC1, cv::Size(r2, r2));
    cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(p, mean_p2, CV_64FC1, cv::Size(r2, r2));

    cv::boxFilter(im.mul(im), corr_I, CV_64FC1, cv::Size(r, r));
    cv::boxFilter(im.mul(im), corr_I2, CV_64FC1, cv::Size(r2, r2));
    cv::boxFilter(im.mul(p), corr_Ip, CV_64FC1, cv::Size(r, r));

    cv::Mat var_I = corr_I - mean_I.mul(mean_I);
    cv::Mat var_I2 = corr_I2 - mean_I2.mul(mean_I2);
    cv::Mat PsiI = ((var_I2 + lamda) * cv::sum(1 / (var_I2 + lamda))[0]) / N;
    cv::Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
    cv::Mat a_psi = cov_Ip / (var_I + eps / PsiI);
    cv::Mat b_psi = mean_p - a_psi.mul(mean_I);

    vector<vector<array<double, 9>>> W = steering_kernel(_im);
    cv::Mat mean_a(im.rows, im.cols, CV_64FC1), mean_b(im.rows, im.cols, CV_64FC1);

    for (int i = 1; i < im.rows - 1; ++i) {
        for (int j = 1; j < im.cols - 1; ++j) {

            cv::Mat Wk(3, 3, CV_64F); // 创建一个3x3的双精度矩阵
            int index = 0; // 用于遍历数组元素的索引
            for (int ii = 0; ii < 3; ++ii) {
                for (int jj = 0; jj < 3; ++jj) {
                    Wk.at<double>(ii, jj) = W[i][j][index]; // 将数组中的元素逐个复制到矩阵中的对应位置
                    index++; // 更新索引
                }
            }
            mean_a.at<double>(i, j) = cv::sum(Wk.mul(a_psi(cv::Rect(j - 1, i - 1, 3, 3))))[0];
            mean_b.at<double>(i, j) = cv::sum(Wk.mul(b_psi(cv::Rect(j - 1, i - 1, 3, 3))))[0];
        }
    }

	a_avg = mean_a;
    mean_b = b_psi;
    cv::Mat qp = mean_a.mul(im) + mean_b;

    return qp;
}

void CLAHE8UC1(const cv::Mat& src, cv::Mat& dst, double clip, cv::Size tiles)
{
	if (src.type() != CV_8UC1)
		return;

	int histSize = 256;
	//图像被划分为tilesX×tilesY个分块
	int tilesX = tiles.width;
	int tilesY = tiles.height;
	cv::Size tileSize;

	cv::Mat srcForLut;
	if (src.size().width % tilesY == 0 && src.size().height % tilesX == 0)
	{
		tileSize = cv::Size(src.size().width / tilesX, src.size().height / tilesY);
		srcForLut = src.clone();
	}
	else
	{
		cv::Mat srcExt;
		//图像边缘填充，保证用于计算LUT的图像其宽度和高度可以分别被tilesX和tilesY整除
		cv::copyMakeBorder(src, srcExt, 0, tilesY - (src.size().height % tilesY), 0, tilesX - (src.size().width % tilesX), cv::BORDER_REFLECT_101);
		tileSize = cv::Size(srcExt.size().width / tilesX, srcExt.size().height / tilesY);
		srcForLut = srcExt.clone();
	}

	//计算直方图“削峰填谷”的阈值，计算clipLimit的过程可以理解为：局部分块图像直方图平均频数的clip倍
	int clipLimit = static_cast<int>(clip * tileSize.area() / histSize);
	clipLimit = std::max(clipLimit, 1);

	cv::Mat lut(tilesX * tilesY, histSize, src.type()); //存储每个分块的LUT，则共有tilesX*tilesY个LUT，每个LUT有histSize个元素
	lut.setTo(0);

	int* tileHist = new int[histSize];

	//对每个块分别计算LUT，存在上面的lut里
	for (int ty = 0; ty < tilesY; ty++)
	{
		for (int tx = 0; tx < tilesX; tx++)
		{
			memset(tileHist, 0, histSize * sizeof(int));
			uchar* tileLut = lut.ptr<uchar>(ty * tilesX + tx);

			cv::Rect tileRect;
			tileRect.x = tx * tileSize.width;
			tileRect.y = ty * tileSize.height;
			tileRect.width = tileSize.width;
			tileRect.height = tileSize.height;

			const cv::Mat tile = srcForLut(tileRect).clone();
			const uchar* pTile = tile.ptr<uchar>();

			//统计局部分块的直方图
			for (int i = 0; i < tile.rows; i++)
			{
				for (int j = 0; j < tile.cols; j++)
				{
					tileHist[pTile[i * tile.cols + j]]++;
				}
			}

			if (clipLimit > 0)
			{
				//直方图“削峰”，并统计削去的像素数
				int clipped = 0;
				for (int i = 0; i < histSize; ++i)
				{
					if (tileHist[i] > clipLimit)
					{
						clipped += (tileHist[i] - clipLimit);
						tileHist[i] = clipLimit;
					}
				}

				//将削去的像素个数平均分配给直方图的每个灰阶
				int redistBatch = clipped / histSize;
				int residual = clipped - redistBatch * histSize;

				//将削去的像素个数平均分配给直方图的每个灰阶，1）首先平均分配
				for (int i = 0; i < histSize; ++i)
					tileHist[i] += redistBatch;

				//将削去的像素个数平均分配给直方图的每个灰阶，2）仍然有剩余的，直方图上等间距分配
				if (residual != 0)
				{
					int residualStep = MAX(histSize / residual, 1);
					for (int i = 0; i < histSize && residual > 0; i += residualStep, residual--)
						tileHist[i]++;
				}
			}


			const float lutScale = cv::saturate_cast<float>(histSize - 1) / (tile.cols * tile.rows);

			//计算LUT
			int sum = 0;
			for (int i = 0; i < histSize; ++i)
			{
				sum += tileHist[i];
				tileLut[i] = cv::saturate_cast<uchar>(sum * lutScale);
			}
		}
	}

	delete[] tileHist;
	tileHist = NULL;

	float invTileW = 1.0f / tileSize.width;
	float invTileH = 1.0f / tileSize.height;

	dst.create(src.size(), CV_8UC1);
	uchar* pDst = dst.ptr<uchar>();
	const uchar* pSrc = src.ptr<uchar>();

	//对每个像素点，找到其4个近邻分块的LUT所映射的像素值，并进行双线性插值
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			uchar pix = pSrc[i * src.cols + j];

			float txf = j * invTileW - 0.5f;

			int tx1 = cvFloor(txf);
			int tx2 = tx1 + 1; //注意这里，说明4个近邻分块是紧挨着的

			float px1 = txf - tx1;
			float px2 = 1.0f - px1;

			tx1 = std::max(tx1, 0);
			tx2 = std::min(tx2, tilesX - 1);

			float tyf = i * invTileH - 0.5f;

			int ty1 = cvFloor(tyf);
			int ty2 = ty1 + 1; //注意这里，说明4个近邻分块是紧挨着的

			float py1 = tyf - ty1;
			float py2 = 1.0f - py1;

			ty1 = std::max(ty1, 0);
			ty2 = std::min(ty2, tilesY - 1);

			uchar* tileLuty1x1 = lut.ptr<uchar>(ty1 * tilesX + tx1);
			uchar* tileLuty1x2 = lut.ptr<uchar>(ty1 * tilesX + tx2);

			uchar* tileLuty2x1 = lut.ptr<uchar>(ty2 * tilesX + tx1);
			uchar* tileLuty2x2 = lut.ptr<uchar>(ty2 * tilesX + tx2);

			//4个邻块的映射灰度值线性插值，先x方向，再y方向
			//注意：前面提到px1+px2=1.0和py1+py2=1.0，这里的px1，px2，py1，py2都是距离。在作为双线性插值的权重时，距离近的权重应大；反之亦然。
			//以x方向插值举例：距离为px1时，其权重应取1.0-px1，即px2；距离为px2时，其权重应取1.0-px2，即px1。
			pDst[i * src.cols + j] = cv::saturate_cast<uchar>(
				(tileLuty1x1[pix] * px2 + tileLuty1x2[pix] * px1) * py2 +
				(tileLuty2x1[pix] * px2 + tileLuty2x2[pix] * px1) * py1);
		}
	}
}
