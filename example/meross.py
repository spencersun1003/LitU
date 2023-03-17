import asyncio
import os
from random import randint

from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager
from meross_iot.model.enums import OnlineStatus

EMAIL = "roimgame@gmail.com"
PASSWORD = "Zbw66714321"

RGB = 255, 247, 199

async def main():
    # Setup the HTTP client API from user-password
    http_api_client = await MerossHttpClient.async_from_user_password(email=EMAIL, password=PASSWORD)

    # Setup and start the device manager
    manager = MerossManager(http_client=http_api_client)
    await manager.async_init()

    # Retrieve the MSL120 devices that are registered on this account
    await manager.async_device_discovery()
    plugs = manager.find_devices(device_type="msl120da", online_status=OnlineStatus.ONLINE)

    if len(plugs) < 1:
        print("No online msl120da smart bulbs found...")
    else:
        # Let's play with RGB colors. Note that not all light devices will support
        # rgb capabilities. For this reason, we first need to check for rgb before issuing
        # color commands.
        dev = plugs[0]

        # Update device status: this is needed only the very first time we play with this device (or if the
        #  connection goes down)
        await dev.async_update()
        if not dev.get_supports_rgb():
            print("Unfortunately, this device does not support RGB...")
        else:
            # Check the current RGB color
            current_color = dev.get_rgb_color()
            print(f"Currently, device {dev.name} is set to color (RGB) = {current_color}")
            # Randomly chose a new color
            print(f"Chosen random color (R,G,B): {RGB}")
            # await dev.async_set_light_color(rgb=RGB, luminance=1)
            await dev.async_set_light_color(onoff=False)
            print("Color changed!")

    # Close the manager and logout from http_api
    manager.close()
    await http_api_client.async_logout()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.stop()