#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class Point {
public:
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

class TreeNode {
public:
    double locationX, locationY;
    TreeNode* parent;
    std::vector<TreeNode*> children;

    TreeNode(double x, double y) : locationX(x), locationY(y), parent(nullptr) {}
    TreeNode& operator= (const TreeNode* other){
        locationX = other->locationX;
        locationY = other->locationY;

        parent = other->parent;

        return *this;
    }
};

class InformedRRTStarAlgorithm {
public:
    InformedRRTStarAlgorithm(const Point& start, const Point& goal, int numIterations, const cv::Mat& grid, double stepSize):
        randomTree(TreeNode(start.x, start.y)),
        start(start),
        goal(TreeNode(goal.x, goal.y)),
        nearestNode(nullptr),
        iterations(std::min(numIterations, 1400)),
        grid(grid),
        rho(stepSize),
        nearestDist(10000),
        // numWaypoints(0),
        searchRadius(rho*2),
        goalArray(goal.x, goal.y),
        goalCosts(10000),
        initialPathFound(false),
        ellipseAngle(atan2(goal.y - start.y, goal.x - start.x)),
        xCenterEllipse(0.5 * (start.x + goal.x)),
        yCenterEllipse(0.5 * (start.y + goal.y)),
        c_min(sqrt((goal.y - start.y) * (goal.y - start.y) + (goal.x - start.x) * (goal.x - start.x))),
        a(generateEllipseAngles()) {}

    void run() {
        cv::Mat display = grid.clone();
        cv::circle(display, cv::Point(goal.locationX, goal.locationY), rho, cv::Scalar(255, 150, 150), cv::FILLED, 1);

        for (int i = 0; i < iterations; ++i) {
            resetNearestValues();
            std::cout<<"Iteration: "<<i+1<<std::endl;

            Point point = sampleAPoint();
            // do {
            //     point = sampleAPoint();
            //     std::cout<<"Working!"<<std::endl;
            // } while (!initialPathFound && !checkIfInEllipse(point.x, point.y, goalCosts.back()));

            double c_best;
            if (initialPathFound){
                c_best = goalCosts.back();

                if (!checkIfInEllipse(point.x, point.y, c_best)) continue;
            }
            // std::cout<<"Working!"<<std::endl;

            findNearest(randomTree, point);
            std::cout << nearestNode->locationX << " " << nearestNode->locationY << std::endl;
            // if (!nearestNode) continue;

            Point newPoint = steerToPoint(*nearestNode, point);
            // std::cout << newPoint.x << " " << newPoint.y << std::endl;
  
            if (!isInObstacle(*nearestNode, newPoint)) {
                findNeighbouringNodes(randomTree, newPoint);
                TreeNode* min_cost_node = nearestNode;
                double min_cost, neighbor_cost;
                min_cost = findPathDistance(min_cost_node) + distance(*nearestNode, newPoint);

                for (TreeNode* neighbor : neighbouringNodes) {
                    neighbor_cost = findPathDistance(neighbor) + distance(*neighbor, newPoint);

                    if (!isInObstacle(*neighbor, newPoint) && neighbor_cost < min_cost){
                        min_cost_node = neighbor;
                        min_cost = neighbor_cost;
                    }
                }

                nearestNode = min_cost_node;
                TreeNode* newNode = new TreeNode(newPoint.x, newPoint.y);
                addChild(newNode);
                
                for (TreeNode* neighbor : neighbouringNodes) {
                    neighbor_cost = min_cost + distance(*neighbor, newPoint);

                    if (!isInObstacle(*neighbor, newPoint) && neighbor_cost < findPathDistance(neighbor)){
                        neighbor->parent = newNode;
                    }
                }
                point = Point(newNode->locationX, newNode->locationY);
                
                if (goalFound(point)){
                    double projectedCost = findPathDistance(newNode) + distance(goal, point);

                    if (projectedCost < goalCosts.back()){
                        initialPathFound = true;
                        addChild(&goal);
                        retracePath();

                        waypoints.insert(waypoints.begin(), start);

                        c_best = goalCosts.back();
                        plotEllipse(c_best);
                    }
                }












                // TreeNode* newNode = new TreeNode(newPoint.x, newPoint.y);
                // addChild(newNode);

                // findNeighbouringNodes(randomTree, newPoint);
                // for (TreeNode* neighbor : neighbouringNodes) {
                //     if (!isInObstacle(*newNode, Point(neighbor->locationX, neighbor->locationY)) &&
                //         findPathDistance(neighbor) + distance(*neighbor, newPoint) < findPathDistance(newNode)) {
                //         newNode->parent->children.erase(std::remove(newNode->parent->children.begin(), newNode->parent->children.end(), newNode),
                //                                         newNode->parent->children.end());
                //         newNode->parent = neighbor;
                //         neighbor->children.push_back(newNode);
                //     }
                // }

                // for (TreeNode* neighbor : neighbouringNodes) {
                //     if (!isInObstacle(*neighbor, newPoint) &&
                //         findPathDistance(newNode) + distance(*neighbor, newPoint) < findPathDistance(neighbor)) {
                //         neighbor->parent->children.erase(std::remove(neighbor->parent->children.begin(), neighbor->parent->children.end(), neighbor),
                //                                         neighbor->parent->children.end());
                //         neighbor->parent = newNode;
                //         newNode->children.push_back(neighbor);
                //     }
                // }

                // if (!initialPathFound) {
                //     if (goalFound(newPoint)) {
                //         addChild(&goal);
                //         retracePath();
                //         initialPathFound = true;
                //     }
                // } else {
                //     if (goalFound(newPoint)) {
                //         addChild(&goal);
                //         retracePath();
                //     }
                //     plotEllipse(goalCosts.back());
                // }

                // // Plot the path (if using OpenCV for visualization)
                // cv::line(display, cv::Point(nearestNode->locationX, nearestNode->locationY),
                // cv::Point(newPoint.x, newPoint.y), cv::Scalar(0, 255, 0), 1);
                // cv::imshow("Informed RRT*", display);
                // cv::waitKey(1);
            }
        }

        for (const Point& waypoint : waypoints) {
            cv::circle(display, cv::Point(waypoint.x, waypoint.y), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Informed RRT*", display);
        cv::waitKey(0);
    }

private:
    // TreeNode randomTree;
    // TreeNode goal;
    // TreeNode* nearestNode;
    // std::vector<TreeNode*> neighbouringNodes;
    // cv::Mat grid;
    // double rho;
    // double nearestDist;
    // int numWaypoints;
    // std::vector<Point> waypoints;
    // std::vector<double> goalCosts = {10000};
    // bool initialPathFound;
    // double ellipseAngle;
    // double xCenterEllipse;
    // double yCenterEllipse;
    // double c_min;
    // std::vector<double> a;
    // int iterations;

    TreeNode randomTree;
    Point start;
    TreeNode goal;
    TreeNode* nearestNode;
    int iterations;
    cv::Mat grid;
    double rho;
    double nearestDist;
    int numWaypoints;
    std::vector<Point> waypoints;
    std::vector<TreeNode*> serializedTree; //not used yet
    int searchRadius;
    std::vector<TreeNode*> neighbouringNodes;
    Point goalArray;
    std::vector<double> goalCosts = {10000};
    bool initialPathFound;
    double ellipseAngle;
    double xCenterEllipse;
    double yCenterEllipse;
    double c_min;
    std::vector<double> a;

    static std::vector<double> generateEllipseAngles() {
        std::vector<double> angles;
        for (double i = 0; i < 2 * M_PI; i += 2 * M_PI / 100) {
            angles.push_back(i);
        }
        return angles;
    }

    Point sampleAPoint() {
        int x = rand() % grid.cols;
        int y = rand() % grid.rows;
        return Point(x, y);
    }

    bool checkIfInEllipse(double pointX, double pointY, double c_best) {
        double rad_x = c_best / 2;
        double rad_y = sqrt(c_best * c_best - c_min * c_min) / 2;
        double transformedX = (pointX - xCenterEllipse) * cos(-ellipseAngle) + (pointY - yCenterEllipse) * sin(-ellipseAngle);
        double transformedY = (pointX - xCenterEllipse) * sin(-ellipseAngle) + (pointY - yCenterEllipse) * cos(-ellipseAngle);
        return ( (transformedX * transformedX) / (rad_x * rad_x) + (transformedY * transformedY) / (rad_y * rad_y)) < 1;
    }

    void plotEllipse(double c_best) {
        double rad_x = c_best / 2;
        double rad_y = sqrt(c_best * c_best - c_min * c_min) / 2;
        std::vector<cv::Point> ellipsePoints;
        for (double angle : a) {
            double x = rad_x * cos(angle) * cos(ellipseAngle) - rad_y * sin(angle) * sin(ellipseAngle) + xCenterEllipse;
            double y = rad_x * cos(angle) * sin(ellipseAngle) + rad_y * sin(angle) * cos(ellipseAngle) + yCenterEllipse;
            ellipsePoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
        }
        cv::polylines(grid, ellipsePoints, true, cv::Scalar(255, 0, 0), 1);
    }

    Point steerToPoint(const TreeNode& start, const Point& end) {
        Point offset(rho * unitVector(start, end).x, rho * unitVector(start, end).y);
        Point point(start.locationX + offset.x, start.locationY + offset.y);
        if (point.x >= grid.cols) point.x = grid.cols - 1;
        if (point.y >= grid.rows) point.y = grid.rows - 1;
        return point;
    }

    bool isInObstacle(const TreeNode& start, const Point& end) {
        Point u_hat = unitVector(start, end);
        Point testPoint(0.0, 0.0);
        int distance = static_cast<int>(this->distance(start, end));
        for (int i = 0; i < distance; ++i) {
            testPoint.x = std::min(static_cast<double>(grid.cols - 1), start.locationX + i * u_hat.x);
            testPoint.y = std::min(static_cast<double>(grid.rows - 1), start.locationY + i * u_hat.y);
            if (grid.at<uchar>(static_cast<int>(round(testPoint.y)), static_cast<int>(round(testPoint.x))) == 1) {
                return true;
            }
        }
        return false;
    }

    Point unitVector(const TreeNode& start, const Point& end) {
        Point v(end.x - start.locationX, end.y - start.locationY);
        double v_norm = sqrt(v.x * v.x + v.y * v.y);
        if (v_norm < 1) v_norm = 1;
        return Point(v.x / v_norm, v.y / v_norm);
    }

    void findNearest(TreeNode& root, const Point& point) {
        // if (root != nullptr) return;

        // std::cout << "HI" << std::endl;

        double dist = distance(root, point);

        if (dist <= nearestDist && root.locationX != goal.locationX) {
            nearestNode = &root;
            nearestDist = dist;
        }

        for (TreeNode* child : root.children) {
            findNearest(*child, point);
        }
    }

    void findNeighbouringNodes(TreeNode& root, const Point& point) {
        // if (root.paren == nullptr) return;

        double dist = distance(root, point);
        if (dist <= rho) {
            neighbouringNodes.push_back(&root);
        }

        for (TreeNode* child : root.children) {
            findNeighbouringNodes(*child, point);
        }
    }

    double distance(const TreeNode& a, const Point& b) const {
        return sqrt((a.locationX - b.x) * (a.locationX - b.x) + (a.locationY - b.y) * (a.locationY - b.y));
    }

    double findPathDistance(TreeNode* node) {
        double costFromRoot = 0;
        TreeNode* currentNode = node;

        while (currentNode->locationX != randomTree.locationX){
            costFromRoot += distance(*currentNode, Point(currentNode->parent->locationX, currentNode->parent->locationY));
            currentNode = currentNode->parent;
        }

        return costFromRoot;
        // double pathDist = 0;
        // while (node->parent != nullptr) {
        //     pathDist += distance(*node, Point(node->parent->locationX, node->parent->locationY));
        //     node = node->parent;
        // }
        // return pathDist;
    }

    void retracePath() {
        numWaypoints = 0;
        waypoints = {};

        double goalCost = 0;
        TreeNode temp_goal(goal);

        while (temp_goal.locationX != randomTree.locationX){
            numWaypoints += 1;
            Point currentPoint = Point(temp_goal.locationX, temp_goal.locationY);
            waypoints.insert(waypoints.begin(), currentPoint);
            goalCost += distance(temp_goal, Point(temp_goal.parent->locationX, temp_goal.parent->locationY));

            temp_goal = temp_goal.parent;
        }

        goalCosts.push_back(goalCost);
    }

    bool goalFound(const Point& point) const {
        return distance(goal, point) < rho;
    }

    void resetNearestValues() {
        nearestNode = nullptr;
        nearestDist = 10000;
        neighbouringNodes = {};
    }

    void addChild(TreeNode* newChild) {
        if (newChild->locationX == goal.locationX){
            nearestNode->children.push_back(&goal);
            goal.parent = nearestNode;
        }
        else{
            nearestNode->children.push_back(newChild);
            newChild->parent = nearestNode;
        }
    }
};

int main() {
    cv::Mat grid = cv::Mat::zeros(1200, 900, CV_8UC3);
    cv::rectangle(grid, cv::Point(0, 0), cv::Point(1200, 900), cv::Scalar(255,255,255), -1);

    //generate random bots

    std::vector<Point> players;
    for(int i=0; i<12; i++){
        players.push_back(Point((std::rand()%1100), (std::rand()%800)));
    }

    int radius = 20;
    for(auto it:players){
        cv::circle(grid, cv::Point(it.x, it.y), radius, cv::Scalar(0, 0, 0), cv::FILLED, 1);
    }

    std::cout<<"working!"<<std::endl;
    Point start(50, 50);
    Point goal(450, 450);

    InformedRRTStarAlgorithm rrt(start, goal, 1000, grid, 30);
    rrt.run();

    return 0;
}
