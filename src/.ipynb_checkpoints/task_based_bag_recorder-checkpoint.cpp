#include <ros/ros.h>
#include <std_msgs/String.h>
#include <jsoncpp/json/json.h>
#include <string>
#include <cstdlib>

class BagRecorder {
private:
    ros::NodeHandle nh;
    ros::Subscriber task_sub;
    std::string current_task_id;
    bool recording;
    pid_t rosbag_pid; // Process ID for rosbag

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
        "/task_executor/task/feedback"
    };

public:
    BagRecorder() : recording(false), rosbag_pid(-1) {
        task_sub = nh.subscribe("/task_executor/task/feedback", 10, &BagRecorder::taskCallback, this);
    }

    void startRecording(const std::string& task_id) {
        if (recording) return;

        current_task_id = task_id;
        std::string bag_filename = "task_" + task_id;
        ROS_INFO_STREAM("Starting bag recording: " << bag_filename << ".bag");

        // Construct the rosbag command
        std::string command = "rosbag record -o /home/jovyan/bags/" + bag_filename;
        for (const auto& topic : recorded_topics) {
            command += " " + topic;
        }
        command += " &"; // Run in the background

        rosbag_pid = std::system(command.c_str());
        recording = true;
    }

    void stopRecording() {
        if (!recording) return;

        ROS_INFO("Stopping bag recording...");
        std::system("pkill -SIGINT -f 'rosbag record'"); // Gracefully stop rosbag
        recording = false;
        current_task_id.clear();
    }

    void taskCallback(const std_msgs::String::ConstPtr& msg) {
        Json::Value task_data;
        Json::Reader reader;

        if (!reader.parse(msg->data, task_data)) {
            ROS_ERROR("Failed to parse JSON.");
            return;
        }

        std::string status = task_data.get("status", "").asString();
        std::string task_id = task_data.get("id", "unknown").asString();

        if (status == "RUNNING") {
            if (current_task_id != task_id) {
                stopRecording();  // Stop any previous recording
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
    BagRecorder recorder;
    ros::spin();
    return 0;
}