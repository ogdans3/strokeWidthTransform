#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#define PI 3.14159265

class Capsule{
    public:
        float x;
        float y;
        Capsule(){};
        Capsule(float x, float y){
            this -> x = x;
            this -> y = y;
        };
};

class Point{
    public:
        cv::Point p;
        float length;
        Point(cv::Point p){
            this -> p = p;
        };
};

std::vector<std::vector<Point> > swt(cv::Mat edges, cv::Mat grad_x, cv::Mat grad_y){
    float distance = 0.05;
    std::vector<std::vector<Point> > rays;
    for(int row = 0; row < edges.size().height; row++){
        uchar* imageData = edges.data;
        const uchar* ptr = (const uchar*)(imageData + row * edges.step);
        for(int col = 0; col < edges.size().width; col++){
            if(*ptr > 0){
//            if(edges.at<uchar>(row, col) > 0){
                std::vector<Point> ray;
                cv::Point p;
                Capsule position;
                cv::Point floored_position;

                position.x = (float) col + 0.5;
                position.y = (float) row + 0.5;
                p.x = col;
                p.y = row;
                ray.push_back(Point(p));

                float gradient_x = grad_x.at<float>(row, col);
                float gradient_y = grad_y.at<float>(row, col);
                // normalize gradient
//                std::cout << row << ", " << col << "  :  " << gradient_x << ", " << gradient_y << "\n";
                float mag = sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y));
                gradient_x = -gradient_x/mag;
                gradient_y = -gradient_y/mag;
                while(true){
                    position.x += distance * gradient_x;
                    position.y += distance * gradient_y;
//                    std::cout << position.x << ", " << position.y << std::endl;
//                    std::cout << gradient_x << ", " << gradient_y << "\n";
//                    std::cout << mag << "\n";
                    floored_position.x = (int)(floor(position.x));
                    floored_position.y = (int)(floor(position.y));

                    //We do not want to process the pixel that we started on
                    if (floored_position.x != p.x || floored_position.y != p.y) {
//                        std::cout << "Not equal" << std::endl;
                        cv::Point candidate(floored_position.x, floored_position.y);
                        //Check if the pixel position is beyond the image boundaries, if so then break. We were unsuccessfull in finding a counter pixel and its probably not text
                        if (candidate.x < 0 || candidate.y < 0 || candidate.x >= edges.size().width || candidate.y >= edges.size().height) {
                            break;
                        }

                        ray.push_back(Point(candidate));

                        //This means that the candidate is not black, and therefore needs further investigation
                        if(edges.at<uchar>(candidate.y, candidate.x) > 0){

                            float c_gradient_x = grad_x.at<float>(candidate.y, candidate.x);
                            float c_gradient_y = grad_y.at<float>(candidate.y, candidate.x);
                            // normalize gradient
                            float c_mag = sqrt((c_gradient_x * c_gradient_x) + (c_gradient_y * c_gradient_y));
//                            std::cout << "C: " << c_gradient_x << ", " << c_gradient_y << ", " << c_mag << "   :::::::  " << candidate.x << ", " << candidate.y << "\n";
                            c_gradient_x = -c_gradient_x/c_mag;
                            c_gradient_y = -c_gradient_y/c_mag;

//                            std::cout << "C2: " << c_gradient_x << ", " << c_gradient_y << "  :  " << gradient_x << ", " << gradient_y << "     : " << acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) << ", " << PI/2 << ", " << (acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) == PI/2) << "\n";
                            //Calculate whether the new gradient is approximately opposite to the old gradient.
                            if (acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) < PI/2.0 ){
                                float length = sqrt((float)((candidate.x - p.x) * (candidate.x - p.x) + (candidate.y - p.y) * (candidate.y - p.y)));
                                std::cout << candidate.x << ", " << candidate.y << "     " << p.x << ", " << p.y << ":  " << length << "\n";
                                for(int i = 0; i < ray.size(); i++) {
                                    ray[i].length = length;
                                }
                                rays.push_back(ray);
                            }
                            break;
                        }
                    }
                }
            }
            ptr++;
        }
    }
    return rays;
}

int main( int argc, char** argv )
{
    std::vector<std::string> paths;
    for(int i = 1; i < argc; i++){
        paths.push_back(std::string(argv[i]));
        if(strstr(argv[i], "-")){
            break;
        }
    }

    for(int i = 0; i < paths.size(); i++){
        std::string filePath = paths[i];

        cv::VideoCapture cap(filePath);

        if (!cap.isOpened()){
            std::cout << "!!! Failed to open file: " << filePath << std::endl;
            return -1;
        }

        cv::Mat frame;
        cv::Mat edges, gray, gaussian;
        cv::Mat grad_x, grad_y, angles;
        for(;;){
            if (!cap.read(frame))
                break;

//            cv::resize(frame, frame, cv::Size(640, 480));
            cv::cvtColor(frame, gray, CV_RGB2GRAY);
            cv::Canny(gray, edges, 175, 320, 3);
            cv::GaussianBlur(gray, gaussian, cv::Size(5, 5), 0, 0);

            cv::Scharr(gaussian, grad_x, CV_32F, 1, 0);
            cv::Scharr(gaussian, grad_y, CV_32F, 0, 1);

            cv::imshow("IMG", frame);
            cv::imshow("GA", gaussian);
            cv::imshow("Edges", edges);
            cv::imshow("grad_x", grad_x);
            cv::imshow("grad_y", grad_y);

            std::vector<std::vector<Point> > rays = swt(edges, grad_x, grad_y);
            std::cout << "Length of rays: " << rays.size() << "\n";
//            exit(0);
            cv::Mat1f swt(edges.size());
            for(int i = 0; i < rays.size(); i++){
                std::cout << "\n";
                for(int q = 0; q < rays[i].size(); q++){
                    Point tmp = rays[i][q];
//                    std::cout << tmp.p << "\n";
//                    std::cout << tmp.p.x << ", " << tmp.p.y << "\n";
                    swt.at<float>(tmp.p) = tmp.length;
                }
            }
            cv::imshow("SWT", swt);

            cv::waitKey(0);
        }
    }
    return 0;
}
