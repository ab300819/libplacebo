gpu_tests = [
  'vulkan.c', 'opengl_surfaceless.c', 'd3d11.c', 'dummy.c', 'metal.c'
]

ts = []

foreach t : tests
  deps = [tdep_static]
  if t == 'opengl_surfaceless.c'
    deps += glad_dep
  endif
  if t == 'vulkan.c'
    deps += vulkan_headers
  endif
  if t == 'metal.c'
    # The backend only passes IOSurfaceRef through to Metal, but this test
    # creates and locks real IOSurfaces and inspects CGColorSpaces, so it needs
    # those frameworks itself.
    deps += dependency('appleframeworks',
                       modules: ['IOSurface', 'CoreFoundation', 'CoreGraphics'])
  endif
  sources = []
  if gpu_tests.contains(t)
    sources += 'gpu_tests.c'
  endif
  ts += { 'test': t,
          'sources': sources,
          'deps': deps }
endforeach

dav1d = dependency('dav1d', required: false)
if dav1d.found()
  ts += { 'test': 'dav1d.c', 'deps': [dav1d, tdep_shared] }
endif

lavu = dependency('libavutil', version: '>=55.74.100', required: false)
lavc = dependency('libavcodec', required: false)
lavf = dependency('libavformat', required: false)
libav_found = lavu.found() and lavc.found() and lavf.found()
if libav_found
  ts += { 'test': 'libav.c', 'deps': [lavu, lavc, lavf, tdep_shared] }
endif

foreach t : ts
  e = executable('test.' + t['test'], [t['test']] + t.get('sources', []),
      objects: t.get('objects', []),
      c_args: [ '-Wno-unused-function' ],
      dependencies: t.get('deps', []),
      link_args: link_args,
      link_depends: link_depends,
  )

  test(t['test'], e, timeout: 120)
endforeach

# Cross-backend comparison test: only when Metal + Vulkan are both enabled
if components.get('metal') and components.get('vulkan')
  cross_be = executable('test.cross_backend.c', 'cross_backend.c',
      c_args: [ '-Wno-unused-function' ],
      dependencies: [tdep_static, vulkan_headers],
      link_args: link_args,
      link_depends: link_depends,
  )
  test('cross_backend.c', cross_be, timeout: 120)
endif

subdir('include')
