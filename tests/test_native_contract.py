import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"robot"))
from native_collector import NativeCollector, CAMERAS
class Props(list):
    def add(self):
        value=SimpleNamespace();self.append(value);return value
class Request:
    START=0;STOP=1
    def __init__(self):self.record_props=Props()
    def SerializeToString(self):raise AssertionError("SDK must receive a message object, not bytes")
class Sdk:
    QoSHistory=SimpleNamespace(kKeepLast="last")
    QoSReliability=SimpleNamespace(kReliable="reliable",kBestEffort="best_effort")
    QoSDurability=SimpleNamespace(kVolatile="volatile")
    def __init__(self):self.published=[];self.readers=[]
    class QoS:pass
    class PubMsgInfo:pass
    def Publisher(self,topic,typ,qos):
        def publish(message,info):
            assert isinstance(message,Request);assert isinstance(info,self.PubMsgInfo)
            self.published.append((topic,message,info));return True
        return SimpleNamespace(publish=publish)
    def Subscriber(self,topic,typ,qos):
        self.readers.append((topic,qos.reliability))
        return SimpleNamespace(subscribe=lambda wait:(b"",None))
class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.sdk=Sdk()
        self.collector=NativeCollector(connect=lambda:(self.sdk,SimpleNamespace(RecordRequest=Request)),topics=["/remote/vr_data","/hal/joint_state","/hal/joint_cmd"])
    def test_sdk_publish_receives_protobuf_and_metadata(self):
        with patch("native_collector.time.sleep"):self.collector._connect()
        props=self.collector.properties("a"*32,Path("/safe/raw"))
        self.collector._publish("a"*32,props,False)
        self.assertEqual(len(self.sdk.published),2)
        self.assertTrue(all(topic.startswith("/dlb_msgs/record_") for topic,_,_ in self.sdk.published))
        self.assertTrue(all(msg.command==Request.START for _,msg,_ in self.sdk.published))
    def test_response_qos_matches_native_best_effort_publishers(self):
        with patch("native_collector.time.sleep"):self.collector._connect()
        self.assertEqual([q for _,q in self.sdk.readers],["best_effort","best_effort"])
    def test_raw_group_and_depth_share_native_video_protocol(self):
        props=self.collector.properties("a"*32,Path("/safe/raw"))
        self.assertEqual({p["key"] for p in props["dds"]},{"/remote/vr_data","/hal/joint_state","/hal/joint_cmd"})
        self.assertEqual({v["key"] for v in props["video"]},set(CAMERAS)|{"head_depth"})
        self.assertNotIn("image",props)
    def test_no_ptp_device_does_not_fabricate_tai_timestamp(self):
        self.collector.proto=SimpleNamespace(RecordRequest=Request)
        message=self.collector.request("a"*32,[{"key":"dds","storage_folder":"/safe/raw/record"}])
        self.assertEqual(message.timestamp_ptp,0);self.assertGreater(message.timestamp_utc,0)
    def test_readiness_does_not_publish_or_open_connections(self):
        self.collector.readiness();self.assertIsNone(self.collector.connections)
        self.assertEqual(self.sdk.published,[])
if __name__=="__main__":unittest.main()
