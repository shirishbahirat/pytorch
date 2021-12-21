-device nvme - ns, iocs = 0x2, zns.zcap = 4096,

nvme list

nvme ns - descs / dev / nvme0n1
nvme ns - descs / dev / nvme0n2

nvme zns id - ns / dev / nvme0n2

nvme zns report - zones / dev / nvme0n2 - s 0x0 - d 3

nvme write / dev / nvme0n2 - s 0x0 - d 3

nvme zns report - zones / dev / nvme0n2 - s 0x0 - d 3

ls / dev/
