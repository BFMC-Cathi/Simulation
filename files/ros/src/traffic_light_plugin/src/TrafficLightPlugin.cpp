#include <iostream>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/gazebo/components/Visual.hh>
#include <ignition/gazebo/components/Pose.hh> // <--- Controls Position
#include <ignition/plugin/Register.hh>

using namespace ignition;
using namespace gazebo;

class TrafficLightPlugin : public System,
                           public ISystemConfigure,
                           public ISystemPreUpdate
{
  public: void Configure(const Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &_ecm,
                         EventManager &/*_eventMgr*/) override
  {
    this->model = Model(_entity);
    this->start_time = std::chrono::steady_clock::now();
    std::cerr << "[TrafficLight] Plugin Configured." << std::endl;
  }

  public: void PreUpdate(const UpdateInfo &_info,
                         EntityComponentManager &_ecm) override
  {
    if (!this->initialized)
    {
      // 1. Find the links
      auto red_link = this->model.LinkByName(_ecm, "red_lens");
      auto yellow_link = this->model.LinkByName(_ecm, "yellow_lens");
      auto green_link = this->model.LinkByName(_ecm, "green_lens");

      // 2. Find the visuals attached to those links
      if (red_link != kNullEntity) this->red_visual = this->FindVisualChild(_ecm, red_link);
      if (yellow_link != kNullEntity) this->yellow_visual = this->FindVisualChild(_ecm, yellow_link);
      if (green_link != kNullEntity) this->green_visual = this->FindVisualChild(_ecm, green_link);

      // 3. Store the "ON" positions (their starting locations)
      if (this->red_visual != kNullEntity && 
          this->yellow_visual != kNullEntity && 
          this->green_visual != kNullEntity) 
      {
        this->red_pose_on = _ecm.Component<components::Pose>(this->red_visual)->Data();
        this->yellow_pose_on = _ecm.Component<components::Pose>(this->yellow_visual)->Data();
        this->green_pose_on = _ecm.Component<components::Pose>(this->green_visual)->Data();

        this->initialized = true;
        std::cerr << "[TrafficLight] SUCCESS! Found all lenses. Starting Magician Cycle (Teleporting)." << std::endl;
      }
      return;
    }

    // Cycle Logic
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count() / 1000.0;

    // Logic: Green (0-5s) -> Yellow (5-7s) -> Red (7-12s)
    if (elapsed < 5.0) {
      this->SetPosition(_ecm, this->green_visual, true, this->green_pose_on);
      this->SetPosition(_ecm, this->yellow_visual, false, this->yellow_pose_on);
      this->SetPosition(_ecm, this->red_visual, false, this->red_pose_on);
    } else if (elapsed < 7.0) {
      this->SetPosition(_ecm, this->green_visual, false, this->green_pose_on);
      this->SetPosition(_ecm, this->yellow_visual, true, this->yellow_pose_on);
      this->SetPosition(_ecm, this->red_visual, false, this->red_pose_on);
    } else if (elapsed < 12.0) {
      this->SetPosition(_ecm, this->green_visual, false, this->green_pose_on);
      this->SetPosition(_ecm, this->yellow_visual, false, this->yellow_pose_on);
      this->SetPosition(_ecm, this->red_visual, true, this->red_pose_on);
    } else {
      this->start_time = std::chrono::steady_clock::now();
    }
  }

  private: Entity FindVisualChild(const EntityComponentManager &_ecm, Entity _parent)
  {
      Entity result = kNullEntity;
      _ecm.Each<components::Visual, components::ParentEntity>(
        [&](const Entity &e, const components::Visual *, const components::ParentEntity *parent) -> bool
        {
            if (parent->Data() == _parent) {
                result = e;
                return false;
            }
            return true;
        });
      return result;
  }

  // THE MAGIC TRICK: If OFF, teleport it 1000 meters underground!
  private: void SetPosition(EntityComponentManager &_ecm, Entity _visual, bool _on, math::Pose3d _original_pose)
  {
    auto poseComp = _ecm.Component<components::Pose>(_visual);
    if (poseComp)
    {
        math::Pose3d current = poseComp->Data();
        math::Pose3d target;

        if (_on) {
            target = _original_pose; // Restore to correct spot
        } else {
            target = _original_pose; 
            target.Pos().Z() -= 1000.0; // Hide underground
        }

        // Only update if position is actually different (to save performance)
        if (current != target) {
            *poseComp = components::Pose(target);
            _ecm.SetChanged(_visual, components::Pose::typeId, ComponentState::OneTimeChange);
        }
    }
  }

  private: Model model;
  private: bool initialized = false;
  private: Entity red_visual = kNullEntity, yellow_visual = kNullEntity, green_visual = kNullEntity;
  private: math::Pose3d red_pose_on, yellow_pose_on, green_pose_on;
  private: std::chrono::steady_clock::time_point start_time;
};

IGNITION_ADD_PLUGIN(TrafficLightPlugin,
                    ignition::gazebo::System,
                    TrafficLightPlugin::ISystemConfigure,
                    TrafficLightPlugin::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(TrafficLightPlugin, "traffic_light_plugin")
