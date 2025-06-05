#include <ros/ros.h>
#include <std_msgs/String.h>
#include <jsoncpp/json/json.h>

#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>

class BagRecorder {
private:
    ros::NodeHandle nh;
    ros::Subscriber task_sub;
    std::string current_task_id;
    bool recording;
    pid_t rosbag_pid; 
    std::string save_path;

    // Topics to record
    std::vector<std::string> recorded_topics = {
        "/realsense_torso_front_camera/color/image_raw_throttle",
        "/realsense_torso_front_camera/color/camera_info",
        "/realsense_front_camera/color/image_raw_throttle",
        "/realsense_front_camera/color/camera_info",
        "/realsense_right_camera/color/image_raw_throttle",
        "/realsense_right_camera/color/camera_info",
        "/realsense_left_camera/color/image_raw_throttle",
        "/realsense_left_camera/color/camera_info",
        "/amcl_pose",
        "/task_executor/task/feedback",
        "/smart_wheel_driver/pdb/imu"
    };

public:
    BagRecorder(ros::NodeHandle& nh) : recording(false), rosbag_pid(-1) {
        task_sub = nh.subscribe("/task_executor/task/feedback", 10, &BagRecorder::taskCallback, this);
        
        if (nh.getParam("save_path", save_path)){
            ROS_INFO("Save path: %s", save_path.c_str());
        } else {
            ROS_WARN("Failed to get save_path param, using default /home/bags");
            save_path = "/home/bags/";
        }
    }

    void startRecording(const std::string& task_id) {
        if (recording) return;

        current_task_id = task_id;
        std::string timestamp = getCurrentTimestamp();
        std::string bag_filename = "task_" + task_id + "_" + timestamp;
        std::string full_path = save_path + "/" + bag_filename;

        ROS_INFO_STREAM("Starting bag recording: " << full_path << ".bag");

        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            std::vector<char*> args;
            args.push_back(strdup("rosbag"));
            args.push_back(strdup("record"));
            args.push_back(strdup(("-o" + full_path).c_str()));

            for (const auto& topic : recorded_topics) {
                args.push_back(strdup(topic.c_str()));
            }

            args.push_back(nullptr); // Null-terminate argv
            execvp("rosbag", args.data());

            // If exec fails
            perror("execvp failed");
            _exit(1);
        } else if (pid > 0) {
            // Parent
            rosbag_pid = pid;
            recording = true;
        } else {
            ROS_ERROR("Failed to fork rosbag process.");
        }
    }


    void stopRecording() {
        if (!recording || rosbag_pid == -1) return;

        ROS_INFO("Stopping bag recording...");

        if (kill(rosbag_pid, SIGINT) == 0) {
            waitpid(rosbag_pid, nullptr, 0);  // Wait for child to exit
        } else {
            perror("Failed to send SIGINT to rosbag process");
        }

        rosbag_pid = -1;
        recording = false;
        current_task_id.clear();
    }


    std::string getCurrentTimestamp() {
        std::time_t now = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&now));
        return std::string(buf);
    }

    void taskCallback(const std_msgs::String::ConstPtr& msg) {
        if (!msg) {
            ROS_WARN_THROTTLE(10, "Received null message pointer.");
            return;
        }

        const std::string& data = msg->data;

        if (data.empty() || data == "''" || data == "\"\"") {
            ROS_WARN_THROTTLE(10, "Received empty task data string.");
            return;
        }

        Json::Value task_data;
        Json::CharReaderBuilder builder;
        builder["collectComments"] = false;
        builder["allowComments"] = false;
        std::string errs;

        std::istringstream s(data);
        if (!Json::parseFromStream(builder, s, &task_data, &errs)) {
            ROS_WARN_STREAM_THROTTLE(10, "Failed to parse task JSON: " << errs);
            return;
        }

        if (!task_data.isMember("status") || !task_data.isMember("id")) {
            ROS_WARN_THROTTLE(10, "Task JSON missing 'status' or 'id'.");
            return;
        }

        std::string status = task_data["status"].asString();
        std::string task_id = task_data["id"].asString();

        if (status == "RUNNING") {
            if (current_task_id != task_id) {
                stopRecording();  // Stop previous
                startRecording(task_id);
            }
        } else {
            stopRecording();
        }
    }


    ~BagRecorder() {
        stopRecording();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "task_based_bag_recorder");
    ros::NodeHandle nh("~");
    BagRecorder recorder(nh);
    ros::spin();
    return 0;
}