metal = get_option('metal')
metal = metal.require(host_machine.system() == 'darwin',
                      error_message: 'Metal requires macOS or iOS')

metal_cpp_use_bundled = false
metal_cpp_found = false
if fs.is_dir('../../3rdparty/metal-cpp')
  # Treat as a system include: metal-cpp headers are not warning-clean under
  # the project's -Wundef and would otherwise break --werror builds.
  metal_cpp_inc = include_directories('../../3rdparty/metal-cpp', is_system: true)
  metal_cpp_found = cxx.has_header('Metal/Metal.hpp',
                                    include_directories: metal_cpp_inc)
  if metal_cpp_found
    metal_cpp_use_bundled = true
  endif
endif
if not metal_cpp_found
  metal_cpp_found = cxx.has_header('Metal/Metal.hpp')
endif
metal = metal.require(metal_cpp_found,
                      error_message: 'metal-cpp headers were not found (expected in 3rdparty/metal-cpp or toolchain includes)')

# -[MTLSharedEvent waitUntilSignaledValue:timeoutMS:] needs macOS 12.0 / iOS
# 15.0. metal-cpp dispatches through objc_msgSend, so too low a deployment
# target compiles fine and then fails at runtime -- check it here.
metal_min_os = cxx.compiles('''
#include <Availability.h>
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED)
# if __MAC_OS_X_VERSION_MIN_REQUIRED < 120000
#  error "Metal backend requires a macOS 12.0 deployment target or later"
# endif
#elif defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
# if __IPHONE_OS_VERSION_MIN_REQUIRED < 150000
#  error "Metal backend requires an iOS 15.0 deployment target or later"
# endif
#else
# error "Unrecognized Apple platform"
#endif
''', name: 'macOS 12.0 / iOS 15.0 deployment target')
metal = metal.require(metal_min_os,
                      error_message: 'Metal requires a macOS 12.0 / iOS 15.0 or later deployment target')

spirv_cross = dependency('spirv-cross-c-shared', version: '>=0.29.0',
                          required: false)
spirv_cross_extra = []
if not spirv_cross.found()
  spirv_cross = dependency('spirv-cross-c', version: '>=0.29.0',
                            required: metal)
  # Static spirv-cross-c needs individual component libraries linked
  spirv_cross_libdir = spirv_cross.get_variable(pkgconfig: 'libdir', default_value: '')
  spirv_cross_dirs = spirv_cross_libdir == '' ? [] : [spirv_cross_libdir]
  spirv_cross_extra = [
    cxx.find_library('spirv-cross-core', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-cpp', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-glsl', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-hlsl', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-msl', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-reflect', dirs: spirv_cross_dirs, required: metal),
    cxx.find_library('spirv-cross-util', dirs: spirv_cross_dirs, required: metal),
  ]
endif

metal_deps = [
  dependency('appleframeworks', modules: ['Metal', 'Foundation', 'QuartzCore', 'CoreGraphics'],
             required: metal),
  spirv_cross,
]
metal_deps += spirv_cross_extra

foreach d : metal_deps
  metal = metal.require(d.found())
endforeach

components.set('metal', metal.allowed())
if metal.allowed()
  if metal_cpp_use_bundled
    build_deps += declare_dependency(dependencies: metal_deps,
                                     include_directories: metal_cpp_inc)
  else
    build_deps += declare_dependency(dependencies: metal_deps)
  endif

  sources += [
    'metal/context.cc',
    'metal/formats.cc',
    'metal/gpu_buf.cc',
    'metal/gpu_pass.cc',
    'metal/gpu_tex.cc',
    'metal/gpu.cc',
    'metal/swapchain.cc',
    'metal/metal_cpp_impl.cc',
  ]
  tests += 'metal.c'
else
  sources += 'metal/stubs.c'
endif
