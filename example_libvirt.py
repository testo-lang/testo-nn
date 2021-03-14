

def make_screenshot(domain):
	stream = domain._conn.newStream()
	domain.screenshot(stream, 0)
	buf = bytearray()
	while True:
		chunk = stream.recv(1000)
		if not chunk:
			break
		buf += chunk
	return buf


# ==========================================

import libvirt
from text_stuff import image_contains_text

conn = libvirt.open('qemu:///system')
vm = conn.lookupByName("VM1")
vm.revertToSnapshot(vm.snapshotLookupByName("initial_state"))
image = make_screenshot(vm)
assert image_contains_text(image, "login:")

