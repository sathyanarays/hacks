import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

discovered_plugins = entry_points(group='myapp.plugins')
print(discovered_plugins)

for plugin in discovered_plugins:
    print(plugin)
    fn = discovered_plugins[plugin.name].load()
    fn()

