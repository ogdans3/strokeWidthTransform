#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/connected_components.hpp>
#include <algorithm>
#include <set>
#include <iterator>

#define PI 3.14159265

class Component{
    public:
        std::vector<cv::Point> points;
        cv::Scalar colorMean;
        cv::Rect rect;
        float meanStrokeWidth;
        bool merged = false;
        cv::Point rectCenter;
        float averageDistanceFromCenter;
        int clusterID;
};
class ComponentCluster{
    public:
        std::vector<Component> cluster;
        void add(Component &cp){
            this -> cluster.push_back(cp);
            cp.merged = true;
        };
        void merge(ComponentCluster &cluster, int clusterID){
            for(int i = 0; i < cluster.cluster.size(); i++){
                this -> add(cluster.cluster[i]);
                cluster.cluster[i].clusterID = clusterID;
            }
        }
};

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
        for(int col = 0; col < edges.size().width; col++){
            if(edges.at<uchar>(row, col) > 0){
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
//                                std::cout << candidate.x << ", " << candidate.y << "     " << p.x << ", " << p.y << ":  " << length << "\n";
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
        }
    }
    return rays;
}

bool PointSort(const Point &lhs, const Point &rhs){
    return lhs.length < rhs.length;
}

void medianFilter(cv::Mat swt, std::vector<std::vector<Point> > rays){
    for(int i = 0; i < rays.size(); i++){
        std::sort(rays[i].begin(), rays[i].end(), &PointSort);
        float median = (rays[i][rays[i].size()/2]).length;
        for(int q = 0; q < rays[i].size(); q++){
            swt.at<float>(rays[i][q].p.y, rays[i][q].p.x) = std::min(rays[i][q].length, median);
        }
    }
}

//Connceted Component algorithm
std::vector<std::vector<cv::Point > > cca(cv::Mat swt){
    float ratio = 3.0;
    int vertices = 0;
    boost::unordered_map<int, int> map;
    boost::unordered_map<int, cv::Point> revmap;

    for(int row = 0; row < swt.size().height; row++){
        for(int col = 0; col < swt.size().width; col++){
            if(swt.at<float>(row, col) > 0.0){
                map[row * swt.size().width + col] = vertices;
//                std::cout << row * swt.size().width << ", " << col << std::endl;
                cv::Point p(col, row);
                revmap[vertices] = p;
                vertices ++;
            }
        }
    }
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> g(vertices);

    int count = 0;
//    std::cout << "\n\n\n\n\n\n" << std::endl;
    for(int row = 0; row < swt.size().height; row++){
        for(int col = 0; col < swt.size().width; col++){
            float mag = swt.at<float>(row, col);
//            std::cout << row << "/" << swt.size().height << ", " << col << "/" << swt.size().width << ", " << mag << "\n";
            if(mag > 0){
                std::vector<cv::Point> neighbours{
                    cv::Point(row + 1, col),
                    cv::Point(row + 1, col - 1),
                    cv::Point(row + 1, col + 1),
                    cv::Point(row, col + 1)
                };
                int pos = map.at(row * swt.size().width + col);
                for(int i = 0; i < neighbours.size(); i++){
                    cv::Point neighbour = neighbours[i];
//                    std::cout << row << ", " << col << "  :::  " << neighbour.x << ", " << neighbour.y << "  :  " << swt.size().width << ", " << swt.size().height << std::endl;
                    if (neighbour.x < 0 || neighbour.y < 0 || neighbour.x >= swt.size().height || neighbour.y >= swt.size().width){
                        continue;
                    }
                    float mag2 = swt.at<float>(neighbour.x, neighbour.y);
//                    std::cout << "Mags: " << mag << ", " << mag2 << "\n";
                    //TODO: Find out if && or || is better, it seems like it depends
                    if(mag2 > 0.0 && (mag / mag2 <= ratio && mag2 / mag <= ratio)){
//                        std::cout << neighbour.x * swt.size().width << ", " << neighbour.y;
//                        std::cout << "   ::::    " << map.at(row * swt.size().width + col) << ", " << map.at(neighbour.x * swt.size().width + neighbour.y);
//                        std::cout << "                        " << row << "/" << swt.size().height << ", " << col << "/" << swt.size().width << "\n";
                        boost::add_edge(pos, map.at(neighbour.x * swt.size().width + neighbour.y), g);
                        count ++;
                    }
                }
            }
        }
    }


    std::vector<int> component (boost::num_vertices (g));
    size_t num_components = boost::connected_components (g, &component[0]);

    std::vector<std::vector<cv::Point > > comps;
    for(int q = 0; q < num_components; q++){
        std::vector<cv::Point> tmpV;
        for (size_t i = 0; i < boost::num_vertices (g); ++i){
            if (component[i] == q){
                tmpV.push_back(revmap[i]);
            }
        }
        comps.push_back(tmpV);
    }
//    std::cout << comps.size() << std::endl;

    return comps;
}

void shit(cv::Mat swt, std::vector<cv::Point> &component,
        float &mean,
        float &variance,
        float &median
){

    std::vector<float> temp;
    mean = 0;
    variance = 0;
    for(int i = 0; i < component.size(); i++){
        float t = swt.at<float>(component[i]);
        mean += t;
        temp.push_back(t);
    }
    mean = mean / (float)(component.size());
    for (std::vector<float>::const_iterator it = temp.begin(); it != temp.end(); it++) {
        variance += (*it - mean) * (*it - mean);
    }
    variance = variance / (float) component.size();
    std::sort(temp.begin(), temp.end());
    median = temp[temp.size()/2];
}

std::vector<Component> filterComponents(cv::Mat swt, std::vector<std::vector<cv::Point> > components){
    std::vector<Component> validComponents;
    for(int i = 0; i < components.size(); i++){
        std::vector<cv::Point> component = components[i];
        cv::Rect rect = cv::boundingRect(component);
        float mean, variance, median;
        shit(swt, component, mean, variance, median);
        Component cp;
        cp.meanStrokeWidth = mean;
        cp.points = component;
        cp.rect = rect;
        cp.rectCenter = cv::Point(((float)cp.rect.x + (float)cp.rect.width)/2, ((float)cp.rect.y + (float)cp.rect.height)/2);
        cp.averageDistanceFromCenter = sqrt((float)cp.rect.width/2 * (float)cp.rect.width/2 + (float)cp.rect.height/2 * (float)cp.rect.height/2);

//        std::cout << variance << ", " << 0.5 * mean << "\n";
//        if (variance > 0.5 * mean)
//            continue;

        if(rect.width < 4 || rect.height < 8)
            continue;

        if(rect.width / rect.height > 10 || rect.height / rect.width > 10)
            continue;

        float diameter = sqrt(rect.width * rect.width + rect.height * rect.height);
        if(diameter / median > 20)
            continue;

        validComponents.push_back(cp);
    }
//    std::cout << "Still valid components: " << validComponents.size() << std::endl;
    return validComponents;
}

std::vector<ComponentCluster> chain(cv::Mat swt, std::vector<Component> &components2, cv::Mat frame){
    float colorThresh = 100;
    float strokeThresh = PI/2;
    float widthThresh = 2.5;
    float heightThresh = 1.5;
    float distanceThresh = 3;
    int minLengthOfCluster = 1;
    std::vector<ComponentCluster> clusters;
    std::vector<Component> components = std::vector<Component>(components2.begin(), components2.end());
    for(int i = 0; i < components.size(); i++){
        Component &cp = components[i];
        cv::Mat roi = frame(cp.rect);
        cv::Mat1b mask(roi.rows, roi.cols);

        cv::Scalar colorMean = cv::mean(roi, mask);
//        std::cout << cp.rect.x << ", " << colorMean << "\n";

        for(int j = 0; j < components.size(); j++){
            if(i == j)
                continue;
            Component &cp2 = components[j];
            cv::Mat roi2 = frame(cp2.rect);
            cv::Mat1b mask2(roi2.rows, roi2.cols);

            cv::Scalar colorMean2 = cv::mean(roi2, mask2);
//            std::cout << colorMean2 << "\n";
//            std::cout << abs(colorMean[0] - colorMean2[0]) << ", ";
//            std::cout << abs(colorMean[1] - colorMean2[1]) << ", ";
//            std::cout << abs(colorMean[2] - colorMean2[2]) << std::endl;
            if( abs(colorMean[0] - colorMean2[0]) > colorThresh ||
                abs(colorMean[1] - colorMean2[1]) > colorThresh ||
                abs(colorMean[2] - colorMean2[2]) > colorThresh)
                continue;

//            std::cout << "Stroke: " << cp.meanStrokeWidth << ", " << cp2.meanStrokeWidth << ",  ";
//            std::cout << std::max(cp.meanStrokeWidth / cp2.meanStrokeWidth, cp2.meanStrokeWidth / cp.meanStrokeWidth) << "\n";
            if( std::max(cp.meanStrokeWidth / cp2.meanStrokeWidth, cp2.meanStrokeWidth / cp.meanStrokeWidth)
                > strokeThresh)
                continue;

            std::cout << std::max((float)cp.rect.height / (float)cp2.rect.height, (float)cp2.rect.height / (float)cp.rect.height) << "\n";
            if( std::max((float)cp.rect.height / (float)cp2.rect.height, (float)cp2.rect.height / (float)cp.rect.height) > heightThresh)
                continue;
            if( (float)cp.rect.width / (float)cp2.rect.width > widthThresh ||
                (float)cp2.rect.width / (float)cp.rect.width > widthThresh)
                continue;
/*
            std::cout << abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y) << ", ";
            std::cout << -cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter << ", ";
            std::cout << abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y)-cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter << ", ";
            std::cout << std::max(cp.averageDistanceFromCenter, cp2.averageDistanceFromCenter) * 3 << ", ";
            std::cout << (abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y)-cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter > std::max(cp.averageDistanceFromCenter, cp2.averageDistanceFromCenter) * 3) << std::endl;
*/
            if(
                abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y) -
                cp.averageDistanceFromCenter -
                cp2.averageDistanceFromCenter
                > distanceThresh)
                continue;

//            std::cout << "Ind: " << i << ", " << j <<  "  :  " "Merged: " << cp.merged << ", " << cp2.merged << "  Cluster size: " << clusters.size() << std::endl;
            if(!cp.merged && !cp2.merged){
                ComponentCluster cluster;
                cluster.add(cp);
                cluster.add(cp2);
                clusters.push_back(cluster);
                cp.clusterID = clusters.size() - 1;
                cp2.clusterID = clusters.size() - 1;
//                std::cout << ", " << "Neither" << std::endl;
            }else if(cp.merged && cp2.merged){
//                std::cout << ", " << "Both" << std::endl;
                if(cp.clusterID == cp2.clusterID){
                    //TODO:This if should probably be at the beginning of the for loop
                    //Alot of unnecessary calculation
//                    std::cout << std::endl;
                    continue;
                }else{
                    ComponentCluster &cluster = clusters[cp.clusterID];
                    ComponentCluster &cluster2 = clusters[cp2.clusterID];
                    cluster.merge(cluster2, cp2.clusterID);
                    clusters.erase(clusters.begin() + cp2.clusterID);
                }
            }else if(cp.merged){
//                std::cout << "First , " << clusters[cp.clusterID].cluster.size() << std::endl;
                ComponentCluster &cluster = clusters[cp.clusterID];
                cluster.add(cp2);
                cp2.clusterID = cp.clusterID;
//                std::cout << cp.merged << ", " << cp2.merged << std::endl;
            }else if(cp2.merged){
//                std::cout << ",    Second, " << clusters.size() << ", " << std::endl;
                ComponentCluster &cluster = clusters[cp2.clusterID];
                cluster.add(cp);
                cp.clusterID = cp2.clusterID;
            }
        }
        if(!cp.merged){
            ComponentCluster cluster;
            cluster.add(cp);
            clusters.push_back(cluster);
            cp.clusterID = clusters.size() - 1;
        }
    }
//    std::cout << "Cluster size: " << clusters.size() << std::endl;
    std::vector<ComponentCluster> finalClusters;
    for(int i = 0; i < clusters.size(); i++){
//        std::cout << "Length of cluster " << i << ": " << clusters[i].cluster.size() << std::endl;
        if(clusters[i].cluster.size() > minLengthOfCluster){
            finalClusters.push_back(clusters[i]);
        }
    }

//    std::cout << "Final number of clusters: " << finalClusters.size() << std::endl;
    return finalClusters;
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
//            std::cout << "Length of rays: " << rays.size() << "\n";
//            exit(0);
            cv::Mat1f swt(edges.size());
            for(int i = 0; i < rays.size(); i++){
//                std::cout << "\n";
                for(int q = 0; q < rays[i].size(); q++){
                    Point tmp = rays[i][q];
//                    std::cout << tmp.p << "\n";
//                    std::cout << tmp.p.x << ", " << tmp.p.y << "\n";
                    swt.at<float>(tmp.p) = tmp.length;
                }
            }
            cv::Mat componentsMat = frame.clone();
            cv::Mat validComponentsMat = frame.clone();
            cv::Mat finalClusterMat = frame.clone();
            cv::imshow("SWT", swt);
            medianFilter(swt, rays);
            std::vector<std::vector<cv::Point> > components = cca(swt);
            std::vector<Component> validComponents = filterComponents(swt, components);
            std::vector<ComponentCluster> clusters = chain(swt, validComponents, frame);

            for(int i = 0; i < components.size(); i++){
                cv::Rect rect = cv::boundingRect(components[i]);
                cv::rectangle(componentsMat, rect, cv::Scalar(0, 255, 255), 2);
                if(i < validComponents.size()){
                    cv::rectangle(validComponentsMat, validComponents[i].rect, cv::Scalar(0, 0, 255), 2);
                }
            }
            for(int i = 0; i < clusters.size(); i++){
                cv::Scalar color = cv::Scalar(rand() % (155) + 100, rand() % 155 + 100, rand() % 155 + 100);
                for(int q = 0; q < clusters[i].cluster.size(); q++){
                    cv::rectangle(finalClusterMat, clusters[i].cluster[q].rect, color, 2);
                }
            }
            cv::imshow("SWTMedian", swt);
            cv::imshow("Components", componentsMat);
            cv::imshow("ValidComponents", validComponentsMat);
            cv::imshow("Final chars", finalClusterMat);

            cv::waitKey(0);
        }
    }
    return 0;
}
