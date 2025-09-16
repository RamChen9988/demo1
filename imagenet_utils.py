import json
import os
import requests
import tempfile
from pathlib import Path

class ImageNetUtils:
    def __init__(self):
        self.class_index = {}
        self.local_json_path = Path("imagenet_class_index_chinese.json")
        
    def download_imagenet_index(self):
        """下载ImageNet类别索引文件"""
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        
        try:
            print("正在下载ImageNet类别索引文件...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 保存到本地文件
            with open(self.local_json_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=2)
            
            print(f"ImageNet类别索引文件已保存到: {self.local_json_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")
            return False
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False
    
    def load_imagenet_index(self):
        """加载ImageNet类别索引"""
        # 首先尝试从本地文件加载
        if self.local_json_path.exists():
            try:
                with open(self.local_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换格式: {"0": ["n01440764", "tench"], ...}
                    self.class_index = data
                    print(f"从本地文件加载了 {len(self.class_index)} 个ImageNet类别")
                    return True
            except Exception as e:
                print(f"加载本地文件失败: {e}")
        
        # 如果本地文件不存在或加载失败，尝试下载
        if self.download_imagenet_index():
            return self.load_imagenet_index()
        
        # 如果下载也失败，使用内置的简化版本作为fallback
        print("使用内置的简化ImageNet类别映射")
        self.class_index = self._get_fallback_index()
        return True
    
    def _get_fallback_index(self):
        """获取内置的简化ImageNet类别映射"""
        return {
            "0": ["n01440764", "tench"],
            "1": ["n01443537", "goldfish"],
            "2": ["n01484850", "great_white_shark"],
            "3": ["n01491361", "tiger_shark"],
            "4": ["n01494475", "hammerhead"],
            "5": ["n01496331", "electric_ray"],
            "6": ["n01498041", "stingray"],
            "7": ["n01514668", "cock"],
            "8": ["n01514859", "hen"],
            "9": ["n01518878", "ostrich"],
            "10": ["n01530575", "brambling"],
            # 动物类别
            "281": ["n02123045", "tabby_cat"],
            "282": ["n02123159", "tiger_cat"],
            "283": ["n02123394", "Persian_cat"],
            "284": ["n02123597", "Siamese_cat"],
            "285": ["n02124075", "Egyptian_cat"],
            "286": ["n02125311", "cougar"],
            "287": ["n02127052", "lynx"],
            "288": ["n02128385", "leopard"],
            "289": ["n02128757", "snow_leopard"],
            "290": ["n02128925", "jaguar"],
            "291": ["n02129165", "lion"],
            "292": ["n02129604", "tiger"],
            "293": ["n02130308", "cheetah"],
            # 交通工具
            "407": ["n03770679", "minivan"],
            "408": ["n03773504", "missile"],
            "409": ["n03775071", "mitten"],
            "435": ["n03888257", "parachute"],
            "436": ["n03888605", "parallel_bars"],
            "437": ["n03891251", "park_bench"],
            "438": ["n03891332", "parking_meter"],
            "445": ["n03930630", "pickup"],
            "446": ["n03933933", "pier"],
            "447": ["n03935335", "piggy_bank"],
            "448": ["n03937543", "pill_bottle"],
            "449": ["n03938244", "pillow"],
            "450": ["n03942813", "ping-pong_ball"],
            "451": ["n03944341", "pinwheel"],
            "452": ["n03947888", "pirate"],
            "453": ["n03950228", "pitcher"],
            "454": ["n03954731", "plane"],
            "455": ["n03956157", "planetarium"],
            "456": ["n03958227", "plastic_bag"],
            "457": ["n03961711", "plate_rack"],
            "458": ["n03967562", "plow"],
            "459": ["n03970156", "plunger"],
            "460": ["n03976467", "Polaroid_camera"],
            "461": ["n03976657", "pole"],
            "462": ["n03977966", "police_van"],
            "463": ["n03980874", "poncho"],
            "464": ["n03982430", "pool_table"],
            "465": ["n03983396", "pop_bottle"],
            "466": ["n03991062", "pot"],
            "467": ["n03992509", "potter's_wheel"],
            "468": ["n03995372", "power_drill"],
            "469": ["n03998194", "prayer_rug"],
            "470": ["n04004767", "printer"],
            "471": ["n04005630", "prison"],
            "472": ["n04008634", "projectile"],
            "473": ["n04009552", "projector"],
            "474": ["n04019541", "puck"],
            "475": ["n04023962", "punching_bag"],
            "476": ["n04026417", "purse"],
            "477": ["n04033901", "quill"],
            "478": ["n04033995", "quilt"],
            "479": ["n04037443", "racer"],
            "480": ["n04039381", "racket"],
            "481": ["n04040759", "radiator"],
            "482": ["n04041544", "radio"],
            "483": ["n04044716", "radio_telescope"],
            "484": ["n04049303", "rain_barrel"],
            "485": ["n04065272", "recreational_vehicle"],
            "486": ["n04067472", "reel"],
            "487": ["n04069434", "reflex_camera"],
            "488": ["n04070727", "refrigerator"],
            "489": ["n04074963", "remote_control"],
            "490": ["n04081281", "restaurant"],
            "491": ["n04086273", "revolver"],
            "492": ["n04090263", "rifle"],
            "493": ["n04099969", "rocking_chair"],
            "494": ["n04111531", "rotisserie"],
            "495": ["n04116512", "rubber_eraser"],
            "496": ["n04118538", "rugby_ball"],
            "497": ["n04118776", "rule"],
            "498": ["n04120489", "running_shoe"],
            "499": ["n04125021", "safe"],
            # 更多常见类别
            "876": ["n04552348", "warplane"],
            "896": ["n04591713", "wine_bottle"],
            "968": ["n07753592", "banana"]
        }
    
    def decode_predictions(self, preds, top=3):
        """解码预测结果"""
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = []
            for i in top_indices:
                class_id = str(i)
                if class_id in self.class_index:
                    class_info = self.class_index[class_id]
                    class_name = class_info[1] if len(class_info) > 1 else f"class_{class_id}"
                else:
                    class_name = f"class_{class_id}"
                result.append((class_id, class_name, pred[i]))
            results.append(result)
        return results
    
    def get_class_name(self, class_id):
        """根据类别ID获取类别名称"""
        class_id_str = str(class_id)
        if class_id_str in self.class_index:
            class_info = self.class_index[class_id_str]
            return class_info[1] if len(class_info) > 1 else f"class_{class_id_str}"
        return f"class_{class_id_str}"

# 全局实例
imagenet_utils = ImageNetUtils()

def load_imagenet_classes():
    """加载ImageNet类别（全局函数）"""
    return imagenet_utils.load_imagenet_index()

def decode_predictions(preds, top=3):
    """解码预测结果（全局函数）"""
    return imagenet_utils.decode_predictions(preds, top)

def get_class_name(class_id):
    """获取类别名称（全局函数）"""
    return imagenet_utils.get_class_name(class_id)
