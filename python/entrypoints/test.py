import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

discovered_plugins = entry_points(group='myapp.plugins')

mod = discovered_plugins['timmins'].load()

mod.hello_world()

mod.hello_world_2()