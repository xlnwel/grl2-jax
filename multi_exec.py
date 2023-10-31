import paramiko
import json
import argparse
import multiprocessing.dummy as mp
import os
import tarfile
import logging


logger = logging.getLogger('multi_exec')
handler = logging.StreamHandler()

if len(logger.handlers) ==  0:
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)
  pass


def pack_working_dir(working_dir: str, output_file: str):
  """
  pack working dir to tar file
  """
  local_filenames = os.listdir(working_dir)

  logger.info('packing: ' + working_dir)
  with tarfile.open(output_file, "w") as tar:
    for filename in local_filenames:
      if filename not in ('.', '..', '.git'):
        logger.debug('packing: ' + filename)
        tar.add(os.path.join(working_dir, filename), arcname=filename)
      pass
    pass
  pass



def process_node(params):
  node_id, node_name, working_dir, conda_activate, \
    ip_address, port, username, password, command, sudo = params

  logger.info('Processing node: {}'.format(node_name))
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(ip_address, port=port, username=username, password=password)  
  # sftp = ssh.open_sftp()
  # sftp.put(working_tar, f'./{working_tar}')
  # sftp.close()

  try:
    # stdin, stdout, stderr = ssh.exec_command(
    #   'mkdir -p ./multi_exec_working_dir && tar -xvf {0} -C ./multi_exec_working_dir'\
    #     .format(working_tar))
    # retcd = stdout.channel.recv_exit_status()
    # if retcd != 0:
    #   logger.info('node: {} failed to extract working dir'.format(node_name))
    #   return False, stdout.read(), stderr.read()

    # if sudo:
    #   cmd = f'cd multi_exec_working_dir && sudo MULTI_EXEC_NODE_NAME={node_name} MULTI_EXEC_NODE_IP={ip_address} {command}'
    # else:
    #   cmd = f'cd multi_exec_working_dir && MULTI_EXEC_NODE_NAME={node_name} MULTI_EXEC_NODE_IP={ip_address} {command}'
    #   pass
    
    if sudo:
      cmd = f'cd {working_dir} && source {conda_activate} grl && sudo NODE_ID={node_id} {command}'
    else:
      cmd = f'cd {working_dir} && source {conda_activate} grl && NODE_ID={node_id} {command}'
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    
    # for sudo
    if sudo:
      stdin.write('{0}\n'.format(password))
      stdin.flush()
      pass
    retcd = stdout.channel.recv_exit_status()
    if retcd != 0:
      logger.info('node: {} failed to execute command'.format(node_name))
      return False, stdout.read(), stderr.read()

    out = stdout.read().decode('utf8')
    err = stderr.read().decode('utf8')

  finally:
    # ssh.exec_command('rm -rf ~/multi_exec_working_dir')
    # ssh.exec_command('rm -rf ~/{0}'.format(working_tar))
    ssh.close()
    pass

  return True, out, err

def main(inventory_file, working_dir, conda_activate, node_range, command, num_workers, sudo):
  with open(inventory_file, 'r') as fin:
    inventory = json.load(fin)
    pass

  nodes = inventory['nodes']

  if node_range is not None:
    start, end = node_range.split(',')
    start = eval(start)
    end = eval(end)
    nodes = {str(i): nodes[str(i)] for i in range(start, end)}

  print('Nodes: {}'.format(','.join([k for k in nodes.keys()])))

#   working_tar = './_multi_exec_working_dir.tar'
#   pack_working_dir(working_dir, working_tar)

  try:
    thread_pool = mp.Pool(num_workers)

    params = [
      (i, k, working_dir, conda_activate, v['ip_address'], v['port'], v['username'], v['password'], command, sudo)
      for i, (k, v) in enumerate(nodes.items())]
    results = thread_pool.imap(process_node, params)

    for r, n in zip(results, [p[0] for p in params]):
      print(f'node {n}, results: [{r[0]}]=========================================>')
      print(f'stdout: {r[1]}')
      print(f'stderr: {r[2]}')
      pass
    thread_pool.close()
  finally:
    # os.remove(working_tar)
    pass
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inventory', type=str, default='inventory.json',
                      help='The place where you store the information of nodes')
  parser.add_argument('--working-dir', '-wd', type=str, default='~',
                      help='The working directory where you execute the command.')
  parser.add_argument('--conda-activate', '-ca', type=str, default='activate', 
                      help='The place where your conda activate is')
  parser.add_argument('--node-range', type=str, default=None, 
                      help='The range of nodes in which you execute the command. Format="start,end"')
  parser.add_argument('--command', type=str, required=True)
  parser.add_argument('--num-workers', type=int, default=8, 
                      help='The number of workers for execution')
  parser.add_argument('--sudo', action='store_true', default=False,
                      help='Executing the command in sudo mode')

  args = parser.parse_args()

  main(args.inventory, args.working_dir, args.conda_activate, args.node_range, args.command, args.num_workers, args.sudo)
