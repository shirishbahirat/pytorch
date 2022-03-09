from paramiko import ServerInterface as server_interface
from threading import Event as event



class dpu_pot(server_interface):

	def __init__(self, client_ip):
		self.client_ip = client_ip
		self.event = event