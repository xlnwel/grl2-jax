# -*- coding:UTF-8 -*-
"""
@FileName：communication_service.py
@Description：与xsim通信交互的底层服务
@Time：2021/6/3 14:15
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
import time
import grpc
from typing import List
from . import HRDataService_pb2 as pb2
from . import HRDataService_pb2_grpc
# import HRDataService_pb2 as pb2
# import HRDataService_pb2_grpc
from .observation_processor import ObservationProcessor
from tools.log import do_logging

import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class CommunicationService(object):
    """
        与XSIM通信交互的底层实现类
    @Examples:
        添加使用示例
		>>> 填写使用说明
		··· 填写简单代码示例
    """
    def __init__(self, address: str):
        # 重置计数器，当重置数量达到100次时，重启当前引擎
        self.address = address
        self.reset_counter = 0
        time.sleep(5)
        self.build_connection()

    def build_connection(self):
        # 创建服务连接
        conn = grpc.insecure_channel(self.address)
        self.client = HRDataService_pb2_grpc.HRDataServiceStub(channel=conn)
        do_logging(f"{self.address},数据服务连接成功！")

    def get_obs(self, domain_id: str = '', engine_id: str = ''):
        """获取态势数据接口"""
        observation = self.client.GetDataObservation(
            pb2.ObservationRequest(
                DomainID=domain_id,
                EngineID=engine_id
            ))

        return observation

    def step(self, cmd_list: List[dict]):
        """引擎推进,发送控制指令接口"""
        cmd_init_entity_control = []
        cmd_line_patrol_control = []
        cmd_area_patrol_control = []
        cmd_change_motion_control = []
        cmd_target_follow_control = []
        cmd_attack_control = []

        for cmd in cmd_list:
            try:
                for k, control in cmd.items():
                    if k == "CmdInitEntityControl":
                        pos = control.get("InitPos")
                        init_pos = pb2.TSVector3dType(
                            X=pos.get("X"),
                            Y=pos.get("Y"),
                            Z=pos.get("Z")
                        )
                        cmd_init_entity_control.append(pb2.CmdInitEntity(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            InitPos=init_pos,
                            InitSpeed=control.get("InitSpeed"),
                            InitHeading=control.get("InitHeading")
                        ))

                    elif k == "CmdLinePatrolControl":
                        coord_list = []
                        for p in control.get("CoordList"):
                            pos = pb2.TSVector3dType(
                                X=p.get("X"),
                                Y=p.get("Y"),
                                Z=p.get("Z")
                            )
                            coord_list.append(pos)

                        cmd_line_patrol_control.append(pb2.CmdLinePatrol(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            CoordList=coord_list,
                            CmdSpeed=control.get("CmdSpeed"),
                            CmdAccMag=control.get("CmdAccMag"),
                            CmdG=control.get("CmdG")
                        ))

                    elif k == "CmdAreaPatrolControl":
                        pos = control.get("CenterCoord")
                        init_pos = pb2.TSVector3dType(
                            X=pos.get("X"),
                            Y=pos.get("Y"),
                            Z=pos.get("Z")
                        )
                        cmd_area_patrol_control.append(pb2.CmdAreaPatrol(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            CenterCoord=init_pos,
                            AreaLength=control.get("AreaLength"),
                            AreaWidth=control.get("AreaWidth"),
                            CmdSpeed=control.get("CmdSpeed"),
                            CmdAccMag=control.get("CmdAccMag"),
                            CmdG=control.get("CmdG")
                        ))

                    elif k == "CmdChangeMotionControl":
                        cmd_change_motion_control.append(pb2.CmdChangeMotion(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            UpdateMotionType=control.get("UpdateMotionType"),
                            CmdSpeed=control.get("CmdSpeed"),
                            CmdAccMag=control.get("CmdAccMag"),
                            CmdG=control.get("CmdG")
                        ))

                    elif k == "CmdTargetFollowControl":
                        cmd_target_follow_control.append(pb2.CmdTargetFollow(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            TgtID=control.get("TgtID"),
                            CmdSpeed=control.get("CmdSpeed"),
                            CmdAccMag=control.get("CmdAccMag"),
                            CmdG=control.get("CmdG")
                        ))

                    elif k == "CmdAttackControl":
                        cmd_attack_control.append(pb2.CmdAttack(
                            HandleID=control.get("HandleID"),
                            Receiver=control.get("Receiver"),
                            TgtID=control.get("TgtID"),
                            Range=control.get("Range")
                        ))

                    else:
                        raise XSimControlError(f" '{k}' 指令无法识别，请严格遵照指令数据格式并通过EnvCmd函数进行组包。")
            except:
                do_logging(f'in communication.step and cmd: {cmd}')
                raise
        flag = True
        for i in range(3):
            try:
                response = self.client.Step(pb2.CmdRequest(
                    CmdInitEntityControl=cmd_init_entity_control,
                    CmdLinePatrolControl=cmd_line_patrol_control,
                    CmdAreaPatrolControl=cmd_area_patrol_control,
                    CmdChangeMotionControl=cmd_change_motion_control,
                    CmdTargetFollowControl=cmd_target_follow_control,
                    CmdAttackControl=cmd_attack_control
                ), timeout=5)
                flag = False
                break
                # logging.info(response)
            except Exception as e:
                do_logging(e)
                do_logging(' ** Xsim Engine Restart ** ')
                do_logging(cmd_list, backtrack=5)
                # do_logging(dict(
                #     CmdInitEntityControl=cmd_init_entity_control,
                #     CmdLinePatrolControl=cmd_line_patrol_control,
                #     CmdAreaPatrolControl=cmd_area_patrol_control,
                #     CmdChangeMotionControl=cmd_change_motion_control,
                #     CmdTargetFollowControl=cmd_target_follow_control,
                #     CmdAttackControl=cmd_attack_control
                # ))
                # raise
                time.sleep(1)
                continue
        if flag:
            raise Exception('step failed')
        return ObservationProcessor.get_obs(self.get_obs())

    def restart(self):
        do_logging("重启XSIM引擎！")
        return self.client.Terminal(pb2.ControlRequest(
            Control="restart"
        ))
    
    def reset(self):
        if self.reset_counter == 100:
            logging.debug("重启XSIM引擎！")
            return self.client.Terminal(pb2.ControlRequest(
                Control="restart"
            ))

        flag = True
        while flag:
            try:
                self.client.Terminal(pb2.ControlRequest(
                    Control="reset"
                ))
                flag = False
            except grpc.RpcError as e:
                do_logging(e)
                flag = True

        return self.client.Terminal(pb2.ControlRequest(
            Control="reset"
        ))

    def close(self):
        return self.client.Terminal(pb2.ControlRequest(
            Control="close"
        ))

    def end(self):
        return self.client.Terminal(pb2.ControlRequest(
            Control="end"
        ))

class ServerError(Exception):
    """数据服务异常基类"""
    pass


class XSimControlError(ServerError):
    """XSim控制指令异常"""
    pass
