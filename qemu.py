-device nvme - ns, iocs = 0x2, zns.zcap = 4096,

nvme list

nvme ns - descs / dev / nvme0n1
nvme ns - descs / dev / nvme0n2

nvme zns id - ns / dev / nvme0n2

nvme zns report - zones / dev / nvme0n2 - s 0x0 - d 3

nvme write / dev / nvme0n2 - s 0x0 - d 3

nvme zns report - zones / dev / nvme0n2 - s 0x0 - d 3

ls / dev/

qemu zns addition

block / nvme.c | 2 + -
hw / block / nvme - ns.c | 276 + ++++++
hw / block / nvme - ns.h | 109 + ++
hw / block / nvme.c | 1615 + +++++++++++++++++++++++++++++++++++++---
hw / block / nvme.h | 8 +
hw / block / trace - events | 32 + -
include / block / nvme.h | 204 + ++++-


qemu - img create - f qcow2 qemucsd.qcow2 4G
qemu - img create - f raw znsssd.img 4G

nvme list
nvme list
nvme list
