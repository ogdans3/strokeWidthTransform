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
#include <unordered_set>
#include <boost/functional/hash.hpp>

#define PI 3.14159265
bool verbose = false;
bool black_on_white = true;

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

std::vector<Point> swt(cv::Mat edges, cv::Mat grad_x, cv::Mat grad_y){
    //TODO: Do thorough tests on the value of distance
    float distance = 0.5;
    std::vector<std::vector<Point> > rays;
    std::vector<Point> allRays;

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
                if(black_on_white){
                    gradient_x = -gradient_x/mag;
                    gradient_y = -gradient_y/mag;
                }else{
                    gradient_x = gradient_x/mag;
                    gradient_y = gradient_y/mag;                    
                }
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
                            if(black_on_white){
                                c_gradient_x = -c_gradient_x/c_mag;
                                c_gradient_y = -c_gradient_y/c_mag;
                            }else{
                                c_gradient_x = c_gradient_x/c_mag;
                                c_gradient_y = c_gradient_y/c_mag;                    
                            }


//                            std::cout << "C2: " << c_gradient_x << ", " << c_gradient_y << "  :  " << gradient_x << ", " << gradient_y << "     : " << acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) << ", " << PI/2 << ", " << (acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) == PI/2) << "\n";
                            //Calculate whether the new gradient is approximately opposite to the old gradient.
                            if (acos(gradient_x * -c_gradient_x + gradient_y * -c_gradient_y) < PI/2 ){
                                float length = sqrt((float)((candidate.x - p.x) * (candidate.x - p.x) + (candidate.y - p.y) * (candidate.y - p.y)));
                                for(int i = 0; i < ray.size(); i++) {
                                    ray[i].length = length;
                                }
                                rays.push_back(ray);
                                allRays.insert(allRays.end(), ray.begin(), ray.end());
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    return allRays;
}

std::vector<std::vector<cv::Point> > mergeCCA( std::vector<std::vector<cv::Point> > connected, 
                                                std::unordered_set <std::pair <int, int>, boost::hash <std::pair <int, int> > > mergeSet
){
    std::vector<int> connectedIndexes(connected.size(), -1);
    std::vector<std::vector<cv::Point> > updatedConnected;
    std::unordered_set <std::pair <int, int>, boost::hash <std::pair <int, int> > > updatedMergeSet;
    for (const std::pair<int, int>& elem: mergeSet) {
        if(connectedIndexes[elem.first] == connectedIndexes[elem.second] && connectedIndexes[elem.first] != -1){
            continue;
        }else if(connectedIndexes[elem.first] == -1 && connectedIndexes[elem.second] == -1){
            std::vector<cv::Point> tmpVector;
            tmpVector.insert(tmpVector.end(), connected[elem.first].begin(), connected[elem.first].end());
            tmpVector.insert(tmpVector.end(), connected[elem.second].begin(), connected[elem.second].end());
            updatedConnected.push_back(tmpVector);
            connectedIndexes[elem.first] = updatedConnected.size() - 1;
            connectedIndexes[elem.second] = updatedConnected.size() - 1;
        }else if(connectedIndexes[elem.first] != connectedIndexes[elem.second] && connectedIndexes[elem.first] != -1 && connectedIndexes[elem.second] != -1){
            updatedMergeSet.insert(std::make_pair(connectedIndexes[elem.first], connectedIndexes[elem.second]));
        }else if(connectedIndexes[elem.first] != -1){
            updatedConnected[connectedIndexes[elem.first]].insert(updatedConnected[connectedIndexes[elem.first]].end(), connected[elem.second].begin(),connected[elem.second].end());
            connectedIndexes[elem.second] = connectedIndexes[elem.first];
        }else if(connectedIndexes[elem.second] != -1){
            updatedConnected[connectedIndexes[elem.second]].insert(updatedConnected[connectedIndexes[elem.second]].end(), connected[elem.first].begin(),connected[elem.first].end());
            connectedIndexes[elem.first] = connectedIndexes[elem.second];
        }
    }

    //TODO: There has to be a more efficient way of merging. This loop should be unnecessary
    for(int i = 0; i < connectedIndexes.size(); i++){
        if(connectedIndexes[i] == -1){
            updatedConnected.push_back(connected[i]);
        }
    }

    if(updatedMergeSet.size() == updatedConnected.size() || updatedMergeSet.size() == 0 || updatedMergeSet.size() == mergeSet.size())
        return updatedConnected;
    return mergeCCA(updatedConnected, updatedMergeSet);
}

//Connceted Component algorithm
std::vector<std::vector<cv::Point > > cca(cv::Mat swt, std::vector<Point> rays){
    long int start = ms();
    float ratio = 3.0;

    boost::unordered_map<int, int> map;
    for(int point = 0; point < rays.size(); point++){
        Point p = rays[point];
        int row = p.p.y;
        int col = p.p.x;
        map[row * swt.size().width + col] = point;
    }

    std::cout << "Graph setup: " << ms() - start << " ms" << std::endl;
    start = ms();

    std::vector<int> indexes(rays.size(), -1);
    int count, count2, count3, count4, count5, count6, count7;
    count = count2 = count3 = count4 = count5 = count6 = count7 = 0;

    std::vector<std::vector<cv::Point> > connected;
    std::unordered_set <std::pair <int, int>, boost::hash <std::pair <int, int> > > mergeSet;

    for(int point = 0; point < rays.size(); point++){
        Point p = rays[point];
        int row = p.p.y;
        int col = p.p.x;

//        float mag = p.length;
        float mag = swt.at<float>(row, col);
//        std::cout << "(" << row << ", " << col << ")" << std::endl;

        int pos = map.at(row * swt.size().width + col);
        std::vector<cv::Point> neighbours{
            cv::Point(row + 1, col),
            cv::Point(row + 1, col - 1),
            cv::Point(row + 1, col + 1),
            cv::Point(row, col + 1)
        };
        for(int i = 0; i < neighbours.size(); i++){
            cv::Point neighbour = neighbours[i];
//            std::cout << row << ", " << col << "  :::  " << neighbour.x << ", " << neighbour.y << "  :  " << swt.size().width << ", " << swt.size().height << std::endl;
            if (neighbour.x < 0 || neighbour.y < 0 || neighbour.x >= swt.size().height || neighbour.y >= swt.size().width){
                continue;
            }
            float mag2 = swt.at<float>(neighbour.x, neighbour.y);
//          std::cout << "Mags: " << mag << ", " << mag2 << "\n";
            //TODO: Find out if && or || is better, it seems like it depends
            count ++;
            if(mag2 > 0.0 && (mag / mag2 <= ratio && mag2 / mag <= ratio)){
                count2 ++;
                int secondPos = map.at(neighbour.x * swt.size().width + neighbour.y);

                if(indexes[pos] == -1 && indexes[secondPos] == -1){
                    count3 ++;
                    std::vector<cv::Point> tmpVec{p.p, cv::Point(neighbour.y, neighbour.x)};
                    connected.push_back(tmpVec);
                    indexes[pos] = connected.size() - 1;
                    indexes[secondPos] = connected.size() - 1;
                }else if(indexes[pos] != indexes[secondPos] && indexes[pos] != -1 && indexes[secondPos] != -1){
                    count4 ++;
                    mergeSet.insert(std::make_pair(indexes[pos], indexes[secondPos]));
                }else if(indexes[pos] == indexes[secondPos] && indexes[pos] != -1){
                    count5 ++;
                    continue;
                }else if(indexes[pos] != -1){
                    count6 ++;
                    connected[indexes[pos]].push_back(cv::Point(neighbour.y, neighbour.x));
                    indexes[secondPos] = indexes[pos];
                }else if(indexes[secondPos] != -1){
                    count7 ++;
                    connected[indexes[secondPos]].push_back(p.p);
                    indexes[pos] = indexes[secondPos];
                }
            }
        }
    }

    std::cout << "CCA Main loop: " << ms() - start << " ms" << std::endl;
    start = ms();

    std::vector<std::vector<cv::Point> > components = mergeCCA(connected, mergeSet);

    std::cout << "CCA merge: " << ms() - start << " ms" << std::endl;
    start = ms();

    std::cout << "CCA Filter zero: " << ms() - start << " ms" << std::endl;
    start = ms();
    if(verbose){
        std::cout << "\t" << "Count: " << count << ", " << count2 << ", " << count3 << ", " << count4 << ", " << count5 << ", " << count6 << ", " << count7 << std::endl;
        std::cout << "\t" << "Connected: " << connected.size() << std::endl;
        std::cout << "\t" << "Unwanted: " << mergeSet.size() << std::endl;
        std::cout << "\t" << "Indexes: " << indexes.size() << std::endl;
        std::cout << "\t"<< "Components: " << components.size() << std::endl;
    }

    return components;
}

void shit(cv::Mat swt, std::vector<cv::Point> &component,
        float &mean,
        float &median
){

    std::vector<float> temp;
    mean = 0;
    for(int i = 0; i < component.size(); i++){
        float t = swt.at<float>(component[i]);
        mean += t;
        temp.push_back(t);
    }
    mean = mean / (float)(component.size());
    std::sort(temp.begin(), temp.end());
    median = temp[temp.size()/2];
}

std::vector<Component> filterComponents(cv::Mat swt, std::vector<std::vector<cv::Point> > components){
    int widthThresh = 4;
    int heightThresh = 8;
    int heightWidthRatioThresh = 10;
    int diameterMedianRatioThresh = 20;

    std::vector<Component> validComponents;
    for(int i = 0; i < components.size(); i++){
        std::vector<cv::Point> component = components[i];
        cv::Rect rect = cv::boundingRect(component);
        float mean, median;
        shit(swt, component, mean, median);
        Component cp;
        cp.meanStrokeWidth = mean;
        cp.points = component;
        cp.rect = rect;
        cp.rectCenter = cv::Point(((float)cp.rect.x + (float)cp.rect.width)/2, ((float)cp.rect.y + (float)cp.rect.height)/2);
        cp.averageDistanceFromCenter = sqrt((float)cp.rect.width/2 * (float)cp.rect.width/2 + (float)cp.rect.height/2 * (float)cp.rect.height/2);

        float diameter = sqrt(rect.width * rect.width + rect.height * rect.height);
        if(verbose){
            std::cout << "\t" << "Median: " << median << std::endl;
            std::cout << "\t" << "Mean: " << mean << std::endl;
            std::cout << "\t" << "Diameter: " << sqrt(rect.width * rect.width + rect.height * rect.height) << std::endl;
            std::cout << std::endl;

            std::cout << "\t" << "Width: " << rect.width << " Cutoff: " << widthThresh << std::endl;
            std::cout << "\t" << "Height: " << rect.height << " Cutoff: " << heightThresh << std::endl;
            std::cout << std::endl;

            std::cout << "\t" << "Width height ratio: " << rect.width / rect.height << std::endl;
            std::cout << "\t" << "Height width ratio: " << rect.height / rect.width << std::endl;
            std::cout << "\t" << "Ration used: " <<  std::max(rect.width / rect.height, rect.height / rect.width) << " Cutoff: " << heightWidthRatioThresh << std::endl;
            std::cout << std::endl;

            std::cout << "\t" << "Diameter over median: " << diameter / median << " Cutoff: " << diameterMedianRatioThresh << std::endl;
            std::cout << std::endl << std::endl;
        }
        if(rect.width < widthThresh || rect.height < heightThresh)
            continue;

        if(std::max(rect.width / rect.height, rect.height / rect.width) > heightWidthRatioThresh)
            continue;

        if(diameter / median > diameterMedianRatioThresh)
            continue;

        validComponents.push_back(cp);
    }
//    std::cout << "Still valid components: " << validComponents.size() << std::endl;
    return validComponents;
}

std::vector<std::vector<Component> > chain(cv::Mat swt, std::vector<Component> &components2, cv::Mat frame){
    float colorThresh = 50;
    float strokeThresh = PI;
    float widthThresh = 2.5;
    float heightThresh = 1.5;
    float distanceThresh = 3;
    int minLengthOfCluster = 1;
//    std::vector<ComponentCluster> clusters;
    std::vector<Component> components = std::vector<Component>(components2.begin(), components2.end());
    std::vector<int> clusterIndexes(components.size(), -1);

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

            if(verbose){
                std::cout << "\t" << colorMean << ", " << colorMean2 << "\n";
                std::cout << "\t" <<    abs(colorMean[0] - colorMean2[0]) +
                                        abs(colorMean[1] - colorMean2[1]) + 
                                        abs(colorMean[2] - colorMean2[2]) << std::endl;
                std::cout << std::endl;
            }
            if( abs(colorMean[0] - colorMean2[0]) +
                abs(colorMean[1] - colorMean2[1]) + 
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

void tmp(cv::Mat frame){
            long int start = ms();
            long int bigBang = ms();

            cv::Mat edges, gray, gaussian;
            cv::Mat grad_x, grad_y, angles;
            cv::Mat componentsMat = frame.clone();
            cv::Mat validComponentsMat = frame.clone();
            cv::Mat finalClusterMat = frame.clone();

            cv::cvtColor(frame, gray, CV_RGB2GRAY);
            cv::Canny(gray, edges, 175, 320, 3);
            cv::GaussianBlur(gray, gaussian, cv::Size(5, 5), 0, 0);

            cv::Scharr(gaussian, grad_x, CV_32F, 1, 0);
            cv::Scharr(gaussian, grad_y, CV_32F, 0, 1);

            std::cout << "Initial image processing (gaussian blur etc.): " << ms() - start << " ms" << "\n";
            start = ms();

            std::vector<Point> rays = swt(edges, grad_x, grad_y);
            std::cout << "SWT: " << ms() - start << " ms" << "\n";
            start = ms();

            cv::Mat swt = cv::Mat::zeros(edges.size().height, edges.size().width, CV_32F);
            std::cout << "\t" << "Ray Size: " << rays.size() << std::endl;
            std::vector<Point> uniqueRays;
            for(int i = 0; i < rays.size(); i++){
                Point tmp = rays[i];
                //Remove duplicate values. the swt is initiated as an empty mat, all zeros, so if the point has a value
                //then we have already visited that point in the rays array, so we delete that point.
                if(swt.at<float>(tmp.p) == 0.0){
                    uniqueRays.push_back(tmp);
                }
                swt.at<float>(tmp.p) = tmp.length;
            }
            std::cout << "\t" << "Unique ray Size: " << uniqueRays.size() << std::endl;
            std::cout << "After SWT: " << ms() - start << " ms" << "\n";
            start = ms();

            std::vector<std::vector<cv::Point> > components = cca(swt, uniqueRays);
            std::cout << "CCA: " << ms() - start << " ms" << "\n";
            start = ms();

            std::vector<Component> validComponents = filterComponents(swt, components);
            std::cout << "Filtered size: " << validComponents.size() << std::endl;
            std::cout << "Filter: " << ms() - start << " ms" << "\n";
            start = ms();

            std::vector<std::vector<Component> > clusters = chain(swt, validComponents, frame);
            std::cout << "Chain: " << ms() - start << " ms" << "\n";
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


            std::cout << "Total time: " << ms() - bigBang << " ms" << "\n";
            cv::imshow("IMG", frame);
            cv::imshow("GA", gaussian);
            cv::imshow("Edges", edges);
            cv::imshow("grad_x", grad_x);
            cv::imshow("grad_y", grad_y);

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

int main(int argc, char** argv){
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
        for(;;){
            if (!cap.read(frame))
                break;

            tmp(frame);
            return 0;
//            cv::resize(frame, frame, cv::Size(640, 480));
        }
    }
    return 0;
}
