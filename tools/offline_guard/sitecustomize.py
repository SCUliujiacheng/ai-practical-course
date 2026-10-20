"""课程进程专用的离线网络限制。允许本机 Jupyter/API，不修改系统网络。"""
import os
if os.environ.get('COURSE_OFFLINE')=='1':
    import socket, ipaddress, json, datetime
    _connect=socket.socket.connect
    _connect_ex=socket.socket.connect_ex
    _getaddrinfo=socket.getaddrinfo
    def allowed(host):
        if isinstance(host,bytes):host=host.decode('ascii','replace')
        if str(host).lower() in ('localhost',socket.gethostname().lower()):return True
        try:return ipaddress.ip_address(str(host).split('%')[0]).is_loopback
        except ValueError:return False
    def reject(address):
        log=os.environ.get('COURSE_NETWORK_LOG')
        if log:
            with open(log,'a',encoding='utf-8') as f:f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'address':str(address)},ensure_ascii=False)+'\n')
        raise OSError('本课程离线模式阻止了外部网络访问：'+str(address))
    def connect(self,address):
        if isinstance(address,tuple) and not allowed(address[0]):reject(address)
        return _connect(self,address)
    def connect_ex(self,address):
        if isinstance(address,tuple) and not allowed(address[0]):reject(address)
        return _connect_ex(self,address)
    def getaddrinfo(host,*args,**kwargs):
        if host is not None and not allowed(host):reject(host)
        return _getaddrinfo(host,*args,**kwargs)
    socket.socket.connect=connect;socket.socket.connect_ex=connect_ex;socket.getaddrinfo=getaddrinfo
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
