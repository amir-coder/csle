from setuptools import setup

setup(name='pycr_common',
      version='0.0.1',
      install_requires=['gym', 'pyglet', 'numpy', 'torch', 'docker', 'paramiko', 'stable_baselines3', 'scp',
                        'random_username', 'jsonpickle', 'Sphinx', 'sphinxcontrib-napoleon',
                        'sphinx-rtd-theme'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='Python Agent Cyber Range Common',
      license='Creative Commons Attribution-ShareAlike 4.0 International',
      keywords='Reinforcement-Learning Cyber-Security',
      url='https://github.com/Limmen/pycr',
      download_url='https://github.com/Limmen/pycr/archive/0.0.1.tar.gz',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: Creative Commons Attribution-ShareAlike 4.0 International',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
  ]
)