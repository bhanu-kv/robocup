#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class TreeNode {
public:
    int locationX;
    int locationY;
    vector<TreeNode*> children;
    TreeNode* parent;

    TreeNode(int x, int y) : locationX(x), locationY(y), parent(nullptr) {}
};

class InformedRRTStarAlgorithm {
public:
    TreeNode* randomTree;
    TreeNode* goal;
    TreeNode* nearestNode;
    int iterations;
    Mat grid;
    int rho;
    double nearestDist;
    int numWaypoints;
    vector<Point> Waypoints;
    vector<vector<int>> serializedTree;
    double searchRadius;
    vector<TreeNode*> neighbouringNodes;
    Point goalArray;
    vector<double> goalCosts;
    bool initialPathFound;
    double ellipseAngle;
    double xCenterEllipse;
    double yCenterEllipse;
    double c_min;
    vector<double> a;

    InformedRRTStarAlgorithm(Point start, Point goal, int numIterations, Mat grid, int stepSize) {
        this->randomTree = new TreeNode(start.x, start.y);
        this->goal = new TreeNode(goal.x, goal.y);
        this->nearestNode = nullptr;
        this->iterations = min(numIterations, 1400);
        this->grid = grid;
        this->rho = stepSize;
        this->nearestDist = 10000;
        this->numWaypoints = 0;
        this->searchRadius = this->rho * 2;
        this->goalArray = goal;
        this->goalCosts.push_back(10000);
        this->initialPathFound = false;
        this->ellipseAngle = atan2(goal.y - start.y, goal.x - start.x);
        this->xCenterEllipse = 0.5 * (start.x + goal.x);
        this->yCenterEllipse = 0.5 * (start.y + goal.y);
        this->c_min = sqrt(pow(goal.y - start.y, 2) + pow(goal.x - start.x, 2));
        this->a = linspace(0, 2 * M_PI, 100);
    }

    void addChild(TreeNode* treeNode) {
        if (treeNode->locationX == goal->locationX) {
            nearestNode->children.push_back(goal);
            goal->parent = nearestNode;
        } else {
            nearestNode->children.push_back(treeNode);
            treeNode->parent = nearestNode;
        }
    }

    Point sampleAPoint() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> disX(1, grid.cols - 1);
        uniform_int_distribution<> disY(1, grid.rows - 1);
        return Point(disX(gen), disY(gen));
    }

    bool checkIfInEllipse(int pointX, int pointY, double c_best) {
        double rad_x = c_best / 2;
        double rad_y = sqrt(c_best * c_best - c_min * c_min) / 2;
        if (((pointX - xCenterEllipse) * cos(-ellipseAngle) + (pointY - yCenterEllipse) * sin(-ellipseAngle)) * ((pointX - xCenterEllipse) * cos(-ellipseAngle) + (pointY - yCenterEllipse) * sin(-ellipseAngle)) / (rad_x * rad_x) + \
            ((pointX - xCenterEllipse) * sin(-ellipseAngle) + (pointY - yCenterEllipse) * cos(-ellipseAngle)) * ((pointX - xCenterEllipse) * sin(-ellipseAngle) + (pointY - yCenterEllipse) * cos(-ellipseAngle)) / (rad_y * rad_y) < 1) {
            return true;
        }
        return false;
    }

    void plotEllipse(double c_best) {
        double rad_x = c_best / 2;
        double rad_y = sqrt(c_best * c_best - c_min * c_min) / 2;
        vector<Point> ellipsePoints;
        for (double angle : a) {
            double x = rad_x * cos(angle) * cos(ellipseAngle) - rad_y * sin(angle) * sin(ellipseAngle) + xCenterEllipse;
            double y = rad_x * cos(angle) * sin(ellipseAngle) - rad_y * sin(angle) * cos(ellipseAngle) + yCenterEllipse;
            ellipsePoints.push_back(Point(x, y));
        }
        for (size_t i = 0; i < ellipsePoints.size() - 1; ++i) {
            line(grid, ellipsePoints[i], ellipsePoints[i + 1], Scalar(0, 0, 255), 1);
        }
    }

    Point steerToPoint(TreeNode* locationStart, Point locationEnd) {
        Point2f offset = rho * unitVector(locationStart, locationEnd);
        Point point(locationStart->locationX + offset.x, locationStart->locationY + offset.y);
        if (point.x >= grid.cols) {
            point.x = grid.cols - 1;
        }
        if (point.y >= grid.rows) {
            point.y = grid.rows - 1;
        }
        return point;
    }

    bool isInObstacle(TreeNode* locationStart, Point locationEnd) {
        Point2f u_hat = unitVector(locationStart, locationEnd);
        Point2f testPoint(0.0, 0.0);
        int distance = static_cast<int>(distanceBetween(locationStart, locationEnd));
        for (int i = 0; i < distance; ++i) {
            testPoint.x = min(grid.cols - 1, locationStart->locationX + i * u_hat.x);
            testPoint.y = min(grid.rows - 1, locationStart->locationY + i * u_hat.y);
            if (grid.at<uchar>(round(testPoint.y), round(testPoint.x)) == 1) {
                return true;
            }
        }
        return false;
    }

    Point2f unitVector(TreeNode* locationStart, Point locationEnd) {
        Point2f v(locationEnd.x - locationStart->locationX, locationEnd.y - locationStart->locationY);
        double v_norm = norm(v);
        if (v_norm < 1) {
            v_norm = 1;
        }
        return v / v_norm;
    }

    void findNearest(TreeNode* root, Point point) {
        if (!root) {
            return;
        }
        double dist = distanceBetween(root, point);
        if (dist <= nearestDist && root->locationX != goal->locationX) {
            nearestNode = root;
            nearestDist = dist;
        }
        for (TreeNode* child : root->children) {
            findNearest(child, point);
        }
    }

    void findNeighbouringNodes(TreeNode* root, Point point) {
        if (!root) {
            return;
        }
        double dist = distanceBetween(root, point);
        if (dist <= searchRadius && dist > 0 && !isInObstacle(root, point)) {
            neighbouringNodes.push_back(root);
        }
        for (TreeNode* child : root->children) {
            findNeighbouringNodes(child, point);
        }
    }

    double distanceBetween(TreeNode* locationStart, Point locationEnd) {
        return sqrt(pow(locationEnd.x - locationStart->locationX, 2) + pow(locationEnd.y - locationStart->locationY, 2));
    }

    bool goalFound(Point locationEnd) {
        double dist = sqrt(pow(goal->locationX - locationEnd.x, 2) + pow(goal->locationY - locationEnd.y, 2));
        if (dist < searchRadius && !isInObstacle(goal, locationEnd)) {
            goalCosts.push_back(findPathDistance(nearestNode) + dist);
            return true;
        }
        return false;
    }

    double findPathDistance(TreeNode* endNode) {
        double cost = 0.0;
        while (endNode->parent) {
            cost += distanceBetween(endNode, Point(endNode->parent->locationX, endNode->parent->locationY));
            endNode = endNode->parent;
        }
        return cost;
    }

    void retracePath() {
        TreeNode* currentNode = goal;
        while (currentNode) {
            Waypoints.push_back(Point(currentNode->locationX, currentNode->locationY));
            currentNode = currentNode->parent;
        }
        reverse(Waypoints.begin(), Waypoints.end());
    }

    void resetNearestValues() {
        nearestNode = nullptr;
        nearestDist = 10000;
        neighbouringNodes.clear();
    }

private:
    vector<double> linspace(double start, double end, int num) {
        vector<double> linspaced;
        double delta = (end - start) / (num - 1);
        for (int i = 0; i < num - 1; ++i) {
            linspaced.push_back(start + delta * i);
        }
        linspaced.push_back(end);
        return linspaced;
    }
};

int main() {
    Mat grid = Mat::zeros(1200, 900, CV_8UC1);

    for (int i = 0; i < 900; ++i) {
        grid.at<uchar>(0, i) = 1;
        grid.at<uchar>(1199, i) = 1;
    }
    for (int i = 0; i < 1200; ++i) {
        grid.at<uchar>(i, 899) = 1;
        grid.at<uchar>(i, 0) = 1;
    }

    Point start(100, 100);
    Point goal(800, 1000);
    int numIterations = 10000;
    int stepSize = 50;

    InformedRRTStarAlgorithm rrt(start, goal, numIterations, grid, stepSize);

    for (int i = 0; i < rrt.iterations; ++i) {
        Point samplePoint = rrt.sampleAPoint();
        if (rrt.initialPathFound) {
            while (!rrt.checkIfInEllipse(samplePoint.x, samplePoint.y, rrt.goalCosts.back())) {
                samplePoint = rrt.sampleAPoint();
            }
        }
        rrt.findNearest(rrt.randomTree, samplePoint);
        Point newPoint = rrt.steerToPoint(rrt.nearestNode, samplePoint);
        if (!rrt.isInObstacle(rrt.nearestNode, newPoint)) {
            TreeNode* newNode = new TreeNode(newPoint.x, newPoint.y);
            rrt.addChild(newNode);
            rrt.findNeighbouringNodes(rrt.randomTree, Point(newNode->locationX, newNode->locationY));
            for (TreeNode* neighbor : rrt.neighbouringNodes) {
                double newNodeCost = rrt.findPathDistance(newNode);
                double neighborCost = rrt.findPathDistance(neighbor);
                double distanceToNewNode = rrt.distanceBetween(neighbor, Point(newNode->locationX, newNode->locationY));
                if (neighborCost + distanceToNewNode < newNodeCost && !rrt.isInObstacle(neighbor, Point(newNode->locationX, newNode->locationY))) {
                    TreeNode* newNodeParent = newNode->parent;
                    auto it = find(newNodeParent->children.begin(), newNodeParent->children.end(), newNode);
                    if (it != newNodeParent->children.end()) {
                        newNodeParent->children.erase(it);
                    }
                    newNode->parent = neighbor;
                    neighbor->children.push_back(newNode);
                }
            }
            for (TreeNode* neighbor : rrt.neighbouringNodes) {
                double newNodeCost = rrt.findPathDistance(newNode);
                double neighborCost = rrt.findPathDistance(neighbor);
                double distanceToNewNode = rrt.distanceBetween(neighbor, Point(newNode->locationX, newNode->locationY));
                if (newNodeCost + distanceToNewNode < neighborCost && !rrt.isInObstacle(newNode, Point(neighbor->locationX, neighbor->locationY))) {
                    TreeNode* neighborParent = neighbor->parent;
                    auto it = find(neighborParent->children.begin(), neighborParent->children.end(), neighbor);
                    if (it != neighborParent->children.end()) {
                        neighborParent->children.erase(it);
                    }
                    neighbor->parent = newNode;
                    newNode->children.push_back(neighbor);
                }
            }
            if (rrt.goalFound(Point(newNode->locationX, newNode->locationY))) {
                rrt.addChild(newNode);
                rrt.retracePath();
                rrt.initialPathFound = true;
            }
        }
        rrt.resetNearestValues();
        if (i % 100 == 0) {
            cout << "Iteration: " << i << endl;
        }
    }

    for (size_t i = 0; i < rrt.Waypoints.size() - 1; ++i) {
        line(grid, rrt.Waypoints[i], rrt.Waypoints[i + 1], Scalar(255), 1);
    }

    namedWindow("Path", WINDOW_AUTOSIZE);
    imshow("Path", grid);
    waitKey(0);

    return 0;
}

