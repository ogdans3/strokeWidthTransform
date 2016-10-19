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
#include <chrono>

#define PI 3.14159265
long int ms(){
    std::chrono::seconds sec(1);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

class Component{
    public:
        std::vector<cv::Point> points;
        cv::Scalar colorMean;
        cv::Rect rect;
        float meanStrokeWidth;
        cv::Point rectCenter;
        float averageDistanceFromCenter;
        int clusterID = -1;
};
typedef std::vector<Component> ComponentCluster;
/*class ComponentCluster{
    public:
        std::vector<Component*> cluster;
        void add(Component &cp){
            std::cout << "Add size before: " << this -> cluster.size() << std::endl;
            this -> cluster.push_back(&cp);
            std::cout << "Add size after: " << this -> cluster.size() << std::endl;
        };
        void merge(ComponentCluster &cluster, int clusterID){
            std::cout << "Sent Cluster ID: " << clusterID << std::endl;
            for(int i = 0; i < cluster.cluster.size(); i++){
                Component &tmpCP = *cluster.cluster[i];
                this -> add(tmpCP);
                std::cout << "Before HERE221312oiu3o213iou12iou3: " << tmpCP.clusterID << std::endl;
                tmpCP.clusterID = clusterID;
                std::cout << "After value HERE221312oiu3o213iou12iou3: " << tmpCP.clusterID << std::endl;
//                std::cout << "After cluster HERE221312oiu3o213iou12iou3: " << cluster.cluster[i].clusterID << std::endl;
            }
        }
};
*/
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
    //TODO: Do thorough tests on the value of distance
    float distance = 0.5;
    std::vector<std::vector<Point> > rays;
    for(int row = 0; row < edges.size().height; row++){
        for(int col = 0; col < edges.size().width; col++){
            if(edges.at<uchar>(row, col) > 0){
                std::vector<Point> ray;
                cv::Point p;
                Capsule position;
                cv::Point floored_position;

                position.x = (float) col;
                position.y = (float) row;
                p.x = col;
                p.y = row;
                ray.push_back(Point(p));

                float gradient_x = grad_x.at<float>(row, col);
                float gradient_y = grad_y.at<float>(row, col);
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
    long int start = ms();
    float ratio = 3.0;
    int vertices = 0;
    boost::unordered_map<int, int> map;
    boost::unordered_map<int, cv::Point> revmap;

    std::cout << "Setup: " << ms() - start << std::endl;
    start = ms();
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
    std::cout << "Graph setup: " << ms() - start << std::endl;
    start = ms();

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
    std::cout << "Main loop: " << ms() - start << std::endl;
    start = ms();


    std::vector<int> component (boost::num_vertices (g));
    size_t num_components = boost::connected_components (g, &component[0]);

    std::vector<std::vector<cv::Point > > comps(num_components);
    for (size_t i = 0; i < boost::num_vertices (g); ++i){
        comps[component[i]].push_back(revmap[i]);
    }
    std::cout << "Connected components: " << ms() - start << std::endl;
    start = ms();

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

std::vector<std::vector<Component> > chain(cv::Mat swt, std::vector<Component> &components2, cv::Mat frame){
    float colorThresh = 100;
    float strokeThresh = PI/2;
    float widthThresh = 2.5;
    float heightThresh = 1.5;
    float distanceThresh = 3;
    int minLengthOfCluster = 1;
//    std::vector<ComponentCluster> clusters;
    std::vector<Component> components = std::vector<Component>(components2.begin(), components2.end());
    std::vector<int> clusterIndexes;
    for(int i = 0; i < components.size(); i++)
        clusterIndexes.push_back(-1);

    std::vector<std::vector<int>> clusters;
    for(int i = 0; i < components.size(); i++){
        Component &cp = components[i];
        cv::Mat roi = frame(cp.rect);
        cv::Mat1b mask(roi.rows, roi.cols);

        cv::Scalar colorMean = cv::mean(roi, mask);
        int &C1ClusterIndex = clusterIndexes[i];
//        std::cout << cp.rect.x << ", " << colorMean << "\n";

        for(int j = i + 1; j < components.size(); j++){
            int &C2ClusterIndex = clusterIndexes[j];
            Component &cp2 = components[j];

//            std::cout << i << ", " << j << ", " << C1ClusterIndex << ", " << C2ClusterIndex << ", " << (C1ClusterIndex == C2ClusterIndex && C2ClusterIndex != -1) << std::endl;
            if(C1ClusterIndex == C2ClusterIndex && C2ClusterIndex != -1 && C1ClusterIndex != -1)
                continue;

            cv::Mat roi2 = frame(cp2.rect);

            cv::Scalar colorMean2 = cv::mean(roi2);
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

//            std::cout << std::max((float)cp.rect.height / (float)cp2.rect.height, (float)cp2.rect.height / (float)cp.rect.height) << "\n";
            if(std::max((float)cp.rect.height / (float)cp2.rect.height, (float)cp2.rect.height / (float)cp.rect.height) > heightThresh)
                continue;
            if(std::max((float)cp.rect.width / (float)cp2.rect.width, (float)cp2.rect.width / (float)cp.rect.width) > widthThresh)
                continue;
/*
            std::cout << abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y) << ", ";
            std::cout << -cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter << ", ";
            std::cout << abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y)-cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter << ", ";
            std::cout << std::max(cp.averageDistanceFromCenter, cp2.averageDistanceFromCenter) * 3 << ", ";
            std::cout << (abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y)-cp.averageDistanceFromCenter-cp2.averageDistanceFromCenter > std::max(cp.averageDistanceFromCenter, cp2.averageDistanceFromCenter) * 3) << std::endl;
*/
            if( abs(cp.rectCenter.x - cp2.rectCenter.x + cp.rectCenter.y - cp.rectCenter.y) -
                cp.averageDistanceFromCenter -
                cp2.averageDistanceFromCenter
                > distanceThresh)
                continue;

            if(C1ClusterIndex == -1 && C2ClusterIndex == -1){
                std::vector<int> cluster{i, j};
                clusters.push_back(cluster);
                C1ClusterIndex = clusters.size() - 1;
                C2ClusterIndex = clusters.size() - 1;
            }else if(C1ClusterIndex != -1 && C2ClusterIndex != -1){
                clusters[C1ClusterIndex].insert(clusters[C1ClusterIndex].end(), clusters[C2ClusterIndex].begin(), clusters[C2ClusterIndex].end());
                int tmp = C2ClusterIndex;
                for(int q = 0; q < clusters[C2ClusterIndex].size(); q++){
                    clusterIndexes[clusters[C2ClusterIndex][q]] = C1ClusterIndex;
                }
                clusters[tmp].empty();
            }else if(C1ClusterIndex != -1){
                clusters[C1ClusterIndex].push_back(j);
                C2ClusterIndex = C1ClusterIndex;
            }else if(C2ClusterIndex != -1){
                clusters[C2ClusterIndex].push_back(i);
                C1ClusterIndex = C2ClusterIndex;
            }
        }
        if(C1ClusterIndex == -1){
            std::vector<int> cluster{i};
            clusters.push_back(cluster);
            C1ClusterIndex = clusters.size() - 1;
        }
    }

//    std::cout << "Cluster size: " << clusters.size() << std::endl;
    std::vector<std::vector<Component> > finalClusters;
    for(int i = 0; i < clusters.size(); i++){
//        std::cout << "Length of cluster " << i << ": " << clusters[i].size() << std::endl;
        if(clusters[i].size() > minLengthOfCluster){
            std::vector<Component> tmp;
            for(int q = 0; q < clusters[i].size(); q++){
                tmp.push_back(components[clusters[i][q]]);
            }
            finalClusters.push_back(tmp);
        }
    }

//    for(int i = 0; i < finalClusters.size(); i++){
//        std::cout << finalClusters[i].size() << std::endl;
//    }

    std::cout << "Final number of chains: " << finalClusters.size() << std::endl;
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

/*
            cv::imshow("IMG", frame);
            cv::imshow("GA", gaussian);
            cv::imshow("Edges", edges);
            cv::imshow("grad_x", grad_x);
            cv::imshow("grad_y", grad_y);
*/

            std::cout << "Starting" << "\n";
            long int start = ms();
            std::vector<std::vector<Point> > rays = swt(edges, grad_x, grad_y);
            std::cout << "SWT: " << ms() - start << "\n";
            start = ms();
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
//            cv::imshow("SWT", swt);
            std::cout << "After SWT: " << ms() - start << "\n";
            start = ms();
            medianFilter(swt, rays);

            std::cout << "Median: " << ms() - start << "\n";
            start = ms();

            std::vector<std::vector<cv::Point> > components = cca(swt);
            std::cout << "CCA: " << ms() - start << "\n";
            start = ms();

            std::vector<Component> validComponents = filterComponents(swt, components);
            std::cout << "Filter: " << ms() - start << "\n";
            start = ms();

            std::vector<std::vector<Component> > clusters = chain(swt, validComponents, frame);
            std::cout << "Chain: " << ms() - start << "\n";
            start = ms();

            for(int i = 0; i < components.size(); i++){
                cv::Rect rect = cv::boundingRect(components[i]);
                cv::rectangle(componentsMat, rect, cv::Scalar(0, 255, 255), 2);
                if(i < validComponents.size()){
                    cv::rectangle(validComponentsMat, validComponents[i].rect, cv::Scalar(0, 0, 255), 2);
                }
            }
            for(int i = 0; i < clusters.size(); i++){
                cv::Scalar color = cv::Scalar(rand() % (155) + 100, rand() % 155 + 100, rand() % 155 + 100);
                for(int q = 0; q < clusters[i].size(); q++){
                    cv::rectangle(finalClusterMat, clusters[i][q].rect, color, 2);
                }
            }
            cv::imshow("SWTMedian", swt);
            cv::imshow("Components", componentsMat);
            cv::imshow("ValidComponents", validComponentsMat);
            cv::imshow("Final chars", finalClusterMat);

/*            cv::Mat finalClusterMatSmall;
            cv::resize(finalClusterMat, finalClusterMatSmall, cv::Size(1280, 720));
            cv::imshow("Final chars small", finalClusterMatSmall);
*/
            cv::waitKey(0);
        }
    }
    return 0;
}
