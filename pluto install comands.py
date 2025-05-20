"""""
Подключите Pluto к хост-машине через USB. Обязательно используйте средний порт USB на Плутоне, потому что другой только для питания. Подключение к Плутону должно создать виртуальный сетевой адаптер, т.е. Плутон выглядит как адаптер USB Ethernet.
На принимающей машине (не ВМ), откройте терминал или предпочитаемый ping инструмент и ping 192.18.2.1. Если это не работает, остановитесь и отладите сетевой интерфейс.
В рамках VM откройте новый терминал
Пинг 192.168.2.1. Если это не сработает, остановитесь здесь и отладитесь. Во время пингинга, отключите свой Плутон и убедитесь, что пинг замедлен, если он продолжает пинговать, то что-то еще по этому IP-адресу находится в сети, и вам придется изменить IP Плутона (или другого устройства), прежде чем двигаться дальше.
Запишите IP-адрес Плутона, потому что он вам понадобится, когда мы начнем использовать Плутон в Python.


sudo apt-get update
sudo apt-get install build-essential git libxml2-dev bison flex libcdk5-dev cmake python3-pip libusb-1.0-0-dev libavahi-client-dev libavahi-common-dev libaio-dev
cd ~
git clone --branch v0.23 https://github.com/analogdevicesinc/libiio.git
cd libiio
mkdir build
cd build
cmake -DPYTHON_BINDINGS=ON ..
make -j$(nproc)
sudo make install
sudo ldconfig

cd ~
git clone https://github.com/analogdevicesinc/libad9361-iio.git
cd libad9361-iio
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install

cd ~
git clone --branch v0.0.14 https://github.com/analogdevicesinc/pyadi-iio.git
cd pyadi-iio
pip3 install --upgrade pip
pip3 install -r requirements.txt
sudo python3 setup.py install
"""