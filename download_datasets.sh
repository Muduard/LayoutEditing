mkdir datasets
cd datasets
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ./
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/
rm annotations_trainval2017.zip val2017.zip train2017.zip
cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
