# roadside_detector
## 概要
画像を入力し、道路端かどうか識別するパッケージ
### node info
```
Node [/roadside_detector_39816_1758801193914]
Publications: 
 * /inter_cls [std_msgs/Int32] ##### 0（道路端以外）or 1（道路端）
 * /rosout [rosgraph_msgs/Log]

Subscriptions: 
 * /camera_center/usb_cam/image_raw [sensor_msgs/Image]
```

## インストール方法
- todo

## 実行方法
### 学習
- rosbagから
1. `roadside_detector/scripts/make_dataset_and_learning.py`内を変更
```
        self.inter_flg = False
        self.ignore_flg = False    
        self.learning_flg = False   ###Falseであることを確認
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flg = False
```
2. `roadside_detector/experiments/play.sh`内を変更
```
rosbag play test.bag    ###ここを再生したいbag名に変更
rosservice call /loop_count "data: true"
```
3. 実行
```
rosrun roadside_detector make_dataset_and_learning.py 
```
```
./play.sh
```
- tensorから
1. `roadside_detector/scripts/make_dataset_and_learning.py`内を変更
```
        self.inter_flg = False
        self.ignore_flg = False    
        self.learning_flg = True   ###Trueであることを確認
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_loop_count)
        self.loop_count_flg = False

        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('roadside_detector') + '/data/'
        self.save_dataset_path = self.path + '/dataset/'
        self.save_model_path = self.path + '/model/'

        self.load_dataset_path = self.save_dataset_path + '/tsukuba' + '/dataset.pt'    ###ここでデータセットを指定
```
2. 実行
```
rosrun roadside_detector make_dataset_and_learning.py 
```
### テスト
```
rosrun roadside_detector test.py 
```
## ライセンス
BSD 3-Clause License