'''
 Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 Description: 考生实现代码
 Note: 缺省代码仅供参考，可自行决定使用、修改或删除
'''

import bisect


class Solution:
    def __init__(self):
        self.allocate_dict = {}
        self.available_ip_set_first = set(range(256))
        self.available_ip_list_first = list(range(256))
        self.available_ip_set_second = set()
        self.available_ip_list_second = list()

    def dhcp_server(self, mac_list):
        data_list = list()

        for value in mac_list:
            d_tmp = value.split('=')
            data_list.append({'action': d_tmp[0], 'mac_list': d_tmp[1]})
        for data in data_list:
            if data['action'] == "REQUEST":
                self.request(data['mac_list'])
            elif data['action'] == "RELEASE":
                self.release(data['mac_list'])

    def request(self, mac):
        if mac in self.allocate_dict:
            if self.allocate_dict[mac][1] == 1:
                print(f"192.168.0.{self.allocate_dict[mac][0]}")
            elif self.allocate_dict[mac][1] == 0 and self.allocate_dict[mac][0] in self.available_ip_set_second:
                print(f"192.168.0.{self.allocate_dict[mac][0]}")
                self.allocate_dict[mac][1] = 1
                self.available_ip_set_second.remove(self.available_ip_list_second[0])
                self.available_ip_list_second.pop(0)
            else:
                if len(self.available_ip_list_first):
                    print(f"192.168.0.{self.available_ip_list_first[0]}")
                    self.allocate_dict[mac] = [self.available_ip_list_first[0], 1]
                    self.available_ip_set_first.remove(self.available_ip_list_first[0])
                    self.available_ip_list_first.pop(0)
                elif len(self.available_ip_list_second):
                    print(f"192.168.0.{self.available_ip_list_second[0]}")
                    self.allocate_dict[mac] = [self.available_ip_list_second[0], 1]
                    self.available_ip_set_second.remove(self.available_ip_list_second[0])
                    self.available_ip_list_second.pop(0)
        elif len(self.available_ip_list_first):
            print(f"192.168.0.{self.available_ip_list_first[0]}")
            self.allocate_dict[mac] = [self.available_ip_list_first[0], 1]
            self.available_ip_set_first.remove(self.available_ip_list_first[0])
            self.available_ip_list_first.pop(0)
        elif len(self.available_ip_list_second):
            print(f"192.168.0.{self.available_ip_list_second[0]}")
            self.allocate_dict[mac] = [self.available_ip_list_second[0], 1]
            self.available_ip_set_second.remove(self.available_ip_list_second[0])
            self.available_ip_list_second.pop(0)
        else:
            print("NA")

    def release(self, mac):
        if mac in self.allocate_dict and self.allocate_dict[mac][1] == 1:
            self.allocate_dict[mac][1] = 0
            self.available_ip_set_second.add(self.allocate_dict[mac][0])
            self.available_ip_list_second.append(self.allocate_dict[mac][0])


if __name__ == "__main__":
    count = int(input().strip())
    mac_list = [input().strip() for _ in range(count)]
    function = Solution()
    function.dhcp_server(mac_list)
