import pymysql
from pymysql import Error
from typing import List, Dict, Optional
import getpass
import sys
import hashlib


class PermissionSystem:
    def __init__(self):
        """初始化数据库连接"""
        try:
            self.connection = pymysql.connect(
                host='localhost',
                user='root',
                password='123456',
                database='my_db',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("\n成功连接到MySQL数据库")
            self._initialize_database()
            if self._should_insert_test_data():
                self._insert_test_data()
            self.current_user = None  # 当前登录用户
        except Error as e:
            print(f"\n错误: 连接MySQL数据库失败: {e}")
            print("请确保:")
            print("1. MySQL服务正在运行")
            print("2. 已创建数据库 'my_db'")
            print("3. 使用 root 用户和密码123456可以访问")
            sys.exit(1)

    def _hash_password(self, password: str) -> str:
        """使用SHA256加密密码"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _should_insert_test_data(self) -> bool:
        """检查是否需要插入测试数据"""
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users")
            (count,) = cursor.fetchone()
            return count == 0

    def _initialize_database(self):
        """确保所有表存在"""
        pass

    def _insert_test_data(self):
        """插入测试数据"""
        print("\n正在初始化测试数据...")
        try:
            with self.connection.cursor() as cursor:
                # 清空所有表数据
                tables = ['user_roles', 'role_permissions', 'feature_permissions',
                          'users', 'roles', 'permissions', 'features']
                for table in tables:
                    cursor.execute(f"DELETE FROM {table}")

                # 插入权限
                permissions = [
                    (1001, '创建用户', 'user:create'),
                    (1002, '编辑用户', 'user:update'),
                    (1003, '删除用户', 'user:delete'),
                    (1004, '查看用户', 'user:view'),
                    (1005, '创建角色', 'role:create'),
                    (1006, '分配角色', 'role:assign'),
                    (1007, '管理权限', 'permission:manage')
                ]
                cursor.executemany("INSERT INTO permissions (id, name, code) VALUES (%s, %s, %s)", permissions)

                # 插入角色
                roles = [
                    (1001, '系统管理员'),
                    (1002, '普通用户'),
                    (1003, '用户管理员')
                ]
                cursor.executemany("INSERT INTO roles (id, name) VALUES (%s, %s)", roles)

                # 插入角色权限关联
                role_permissions = [
                    (1001, 1001), (1001, 1002), (1001, 1003), (1001, 1004),
                    (1001, 1005), (1001, 1006), (1001, 1007),
                    (1003, 1001), (1003, 1002), (1003, 1004),
                    (1002, 1004)
                ]
                cursor.executemany("INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)",
                                   role_permissions)

                # 插入功能
                features = [
                    (1, '用户管理', 'user_management'),
                    (2, '角色管理', 'role_management'),
                    (3, '权限管理', 'permission_management')
                ]
                cursor.executemany("INSERT INTO features (id, name, code) VALUES (%s, %s, %s)", features)

                # 插入功能权限关联
                feature_permissions = [
                    (1, 1001), (1, 1002), (1, 1003), (1, 1004),
                    (2, 1005), (2, 1006),
                    (3, 1007)
                ]
                cursor.executemany("INSERT INTO feature_permissions (feature_id, permission_id) VALUES (%s, %s)",
                                   feature_permissions)

                # 插入用户(密码使用SHA256加密)
                users = [
                    (1001, 'admin', self._hash_password('admin123')),
                    (1002, 'user_manager', self._hash_password('manager123')),
                    (1003, 'normal_user', self._hash_password('user123'))
                ]
                cursor.executemany("INSERT INTO users (id, username, password) VALUES (%s, %s, %s)", users)

                # 插入用户角色关联
                user_roles = [
                    (1001, 1001),
                    (1002, 1003),
                    (1003, 1002)
                ]
                cursor.executemany("INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s)", user_roles)

            self.connection.commit()
            print("\n测试数据初始化完成")
            print("默认管理员账号: admin / admin123")
            print("用户管理员账号: user_manager / manager123")
            print("普通用户账号: normal_user / user123")
        except Error as e:
            print(f"\n错误: 初始化测试数据失败: {e}")
            self.connection.rollback()
            sys.exit(1)

    # 登录与注册功能
    def login(self, username: str, password: str) -> bool:
        """用户登录"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

                if user and user['password'] == password:
                    self.current_user = user
                    print(f"\n✓ 登录成功，欢迎 {username}!")
                    return True
                else:
                    print("\n✗ 登录失败: 用户名或密码错误")
                    return False
        except Error as e:
            print(f"\n✗ 登录失败: {e}")
            return False

    def register(self, username: str, password: str) -> bool:
        """用户注册"""


        try:
            with self.connection.cursor() as cursor:
                # 检查用户名是否已存在
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    print("\n✗ 注册失败: 用户名已存在")
                    return False

                # 创建新用户
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    (username,password)
                )

                # 默认分配普通用户角色
                cursor.execute("SELECT id FROM roles WHERE name = '普通用户'")
                role = cursor.fetchone()
                if role:
                    cursor.execute(
                        "INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s)",
                        (cursor.lastrowid, role['id'])
                    )

            self.connection.commit()
            print(f"\n✓ 用户 {username} 注册成功")
            return True
        except Error as e:
            print(f"\n✗ 注册失败: {e}")
            self.connection.rollback()
            return False

    def logout(self):
        """用户登出"""
        if self.current_user:
            print(f"\n✓ 用户 {self.current_user['username']} 已登出")
            self.current_user = None
        else:
            print("\n✗ 当前没有登录的用户")

    # 权限检查功能
    # def _check_permission(self, permission_code: str) -> bool:
    #     """检查当前用户是否有指定权限"""
    #     if not self.current_user:
    #         return False
    #
    #     try:
    #         with self.connection.cursor() as cursor:
    #             cursor.execute("""
    #                 SELECT COUNT(*) FROM permissions p
    #                 JOIN role_permissions rp ON p.id = rp.permission_id
    #                 JOIN user_roles ur ON rp.role_id = ur.role_id
    #                 WHERE ur.user_id = %s AND p.code = %s
    #             """, (self.current_user['id'], permission_code))
    #             (count,) = cursor.fetchone()
    #             return count > 0
    #     except Error as e:
    #         print(f"\n✗ 权限检查失败: {e}")
    #         return False

    def _check_permission(self, permission_code: str) -> bool:
        """检查当前用户是否有指定权限"""
        if not self.current_user:
            print("\n✗ 请先登录")
            return False

        try:
            with self.connection.cursor() as cursor:
                # 修改查询，使用COUNT(1)并明确获取结果
                cursor.execute("""
                    SELECT COUNT(1) as permission_count 
                    FROM permissions p
                    JOIN role_permissions rp ON p.id = rp.permission_id
                    JOIN user_roles ur ON rp.role_id = ur.role_id
                    WHERE ur.user_id = %s AND p.code = %s
                """, (self.current_user['id'], permission_code))

                result = cursor.fetchone()
                # 确保转换为整数
                count = int(result['permission_count']) if result else 0
                return count > 0

        except Error as e:
            print(f"\n✗ 权限检查失败: {e}")
            return False

    # 用户管理功能(添加权限检查)
    def create_user(self, username: str, password: str) -> bool:
        """创建新用户"""
        if not self._check_permission('user:create'):
            print("\n✗ 错误: 没有创建用户的权限")
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    (username, password))
                self.connection.commit()
                print(f"\n✓ 用户 {username} 创建成功")
            return True
        except Error as e:
            print(f"\n✗ 错误: 创建用户失败: {e}")
            self.connection.rollback()
            return False

    def get_users(self) -> List[Dict]:
        """获取所有用户列表"""
        if not self._check_permission('user:view'):
            print("\n✗ 错误: 没有查看用户的权限")
            return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users ORDER BY id")
                return cursor.fetchall()
        except Error as e:
            print(f"\n✗ 错误: 获取用户列表失败: {e}")
            return []

    def delete_user(self, user_id: int) -> bool:
        """删除用户"""
        if not self._check_permission('user:delete'):
            print("\n✗ 错误: 没有删除用户的权限")
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                affected = cursor.rowcount
            self.connection.commit()
            if affected > 0:
                print(f"\n✓ 用户ID {user_id} 删除成功")
                return True
            print(f"\n✗ 错误: 未找到用户ID {user_id}")
            return False
        except Error as e:
            print(f"\n✗ 错误: 删除用户失败: {e}")
            self.connection.rollback()
            return False

    # 角色管理功能(添加权限检查)
    def create_role(self, role_name: str) -> bool:
        """创建新角色"""
        if not self._check_permission('role:create'):
            print("\n✗ 错误: 没有创建角色的权限")
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("INSERT INTO roles (name) VALUES (%s)", (role_name,))
            self.connection.commit()
            print(f"\n✓ 角色 '{role_name}' 创建成功")
            return True
        except Error as e:
            print(f"\n✗ 错误: 创建角色失败: {e}")
            self.connection.rollback()
            return False

    def get_roles(self) -> List[Dict]:
        """获取所有角色列表"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM roles ORDER BY id")
                return cursor.fetchall()
        except Error as e:
            print(f"\n✗ 错误: 获取角色列表失败: {e}")
            return []

    def assign_role_to_user(self, user_id: int, role_id: int) -> bool:
        """为用户分配角色"""
        if not self._check_permission('role:assign'):
            print("\n✗ 错误: 没有分配角色的权限")
            return False

        try:
            with self.connection.cursor() as cursor:
                # 检查用户和角色是否存在
                cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                if not cursor.fetchone():
                    print(f"\n✗ 错误: 用户ID {user_id} 不存在")
                    return False

                cursor.execute("SELECT id FROM roles WHERE id = %s", (role_id,))
                if not cursor.fetchone():
                    print(f"\n✗ 错误: 角色ID {role_id} 不存在")
                    return False

                cursor.execute("INSERT INTO user_roles (user_id, role_id) VALUES (%s, %s)",
                               (user_id, role_id))
            self.connection.commit()
            print(f"\n✓ 为用户ID {user_id} 分配角色ID {role_id} 成功")
            return True
        except Error as e:
            print(f"\n✗ 错误: 分配角色失败: {e}")
            self.connection.rollback()
            return False

    # 权限管理功能(添加权限检查)
    def get_permissions(self) -> List[Dict]:
        """获取所有权限列表"""
        if not self._check_permission('permission:manage'):
            print("\n✗ 错误: 没有管理权限的权限")
            return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM permissions ORDER BY id")
                return cursor.fetchall()
        except Error as e:
            print(f"\n✗ 错误: 获取权限列表失败: {e}")
            return []

    def add_permission_to_role(self, role_id: int, permission_id: int) -> bool:
        """为角色添加权限"""
        if not self._check_permission('permission:manage'):
            print("\n✗ 错误: 没有管理权限的权限")
            return False

        try:
            with self.connection.cursor() as cursor:
                # 检查角色和权限是否存在
                cursor.execute("SELECT id FROM roles WHERE id = %s", (role_id,))
                if not cursor.fetchone():
                    print(f"\n✗ 错误: 角色ID {role_id} 不存在")
                    return False

                cursor.execute("SELECT id FROM permissions WHERE id = %s", (permission_id,))
                if not cursor.fetchone():
                    print(f"\n✗ 错误: 权限ID {permission_id} 不存在")
                    return False

                cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)",
                               (role_id, permission_id))
            self.connection.commit()
            print(f"\n✓ 为角色ID {role_id} 添加权限ID {permission_id} 成功")
            return True
        except Error as e:
            print(f"\n✗ 错误: 添加权限失败: {e}")
            self.connection.rollback()
            return False

    def get_user_permissions(self, user_id: int) -> List[Dict]:
        """获取用户的所有权限"""
        if not self._check_permission('user:view'):
            print("\n✗ 错误: 没有查看用户权限的权限")
            return []

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT p.* FROM permissions p
                    JOIN role_permissions rp ON p.id = rp.permission_id
                    JOIN user_roles ur ON rp.role_id = ur.role_id
                    WHERE ur.user_id = %s
                    ORDER BY p.id
                """, (user_id,))
                return cursor.fetchall()
        except Error as e:
            print(f"\n✗ 错误: 获取用户权限失败: {e}")
            return []

    # def check_permission(self, user_id: int, permission_code: str) -> bool:
    #     """检查用户是否拥有特定权限"""
    #     try:
    #         with self.connection.cursor() as cursor:
    #             cursor.execute("""
    #                 SELECT COUNT(*) FROM permissions p
    #                 JOIN role_permissions rp ON p.id = rp.permission_id
    #                 JOIN user_roles ur ON rp.role_id = ur.role_id
    #                 WHERE ur.user_id = %s AND p.code = %s
    #             """, (user_id, permission_code))
    #             (count,) = cursor.fetchone()
    #             return count > 0
    #     except Error as e:
    #         print(f"\n✗ 错误: 检查权限失败: {e}")
    #         return False


def check_permission(self, user_id: int, permission_code: str) -> bool:
    """检查指定用户是否有特定权限"""
    try:
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(1) as permission_count
                FROM permissions p
                JOIN role_permissions rp ON p.id = rp.permission_id
                JOIN user_roles ur ON rp.role_id = ur.role_id
                WHERE ur.user_id = %s AND p.code = %s
            """, (user_id, permission_code))

            result = cursor.fetchone()
            # 确保转换为整数
            count = int(result['permission_count']) if result else 0
            return count > 0

    except Error as e:
        print(f"\n✗ 检查权限失败: {e}")
        return False



def display_menu(title: str, options: List[str]) -> str:
    """显示菜单并获取用户选择"""
    print(f"\n=== {title} ===")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"{len(options) + 1}. 返回")

    while True:
        choice = input("请选择操作: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options) + 1:
            return choice
        print("无效输入，请输入数字选项")


def display_table(headers: List[str], rows: List[Dict], columns: List[str]):
    """显示表格数据"""
    if not rows:
        print("\n没有数据可显示")
        return

    # 计算每列最大宽度
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, col in enumerate(columns):
            col_widths[i] = max(col_widths[i], len(str(row.get(col, ''))))

    # 打印表头
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    separator = "-" * len(header_line)
    print(f"\n{separator}\n{header_line}\n{separator}")

    # 打印数据行
    for row in rows:
        print(" | ".join(str(row.get(col, '')).ljust(col_widths[i]) for i, col in enumerate(columns)))

    print(separator)


def auth_menu(system: PermissionSystem):
    """认证菜单(登录/注册)"""
    while True:
        choice = display_menu("认证菜单", [
            "登录",
            "注册",
            "退出系统"
        ])

        if choice == '1':  # 登录
            username = input("用户名: ")
            password = input("密码: ")
            if system.login(username, password):
                return True

        elif choice == '2':  # 注册
            username = input("用户名: ")
            password = input("密码: ")
            confirm = input("确认密码: ")
            if password == confirm:
                system.register(username, password)
            else:
                print("\n✗ 两次输入的密码不一致")

        elif choice == '3':  # 退出
            return False


def user_management_menu(system: PermissionSystem):
    """用户管理菜单"""
    while True:
        choice = display_menu("用户管理", [
            "列出所有用户",
            "创建新用户",
            "删除用户",
            "为用户分配角色",
            "查看用户权限",
            "查看我的权限"
        ])

        if choice == '1':  # 列出所有用户
            users = system.get_users()
            display_table(["ID", "用户名"], users, ["id", "username"])

        elif choice == '2':  # 创建新用户
            print("\n创建新用户")
            username = input("用户名: ")
            password = getpass.getpass("密码: ")
            system.create_user(username, password)

        elif choice == '3':  # 删除用户
            user_id = input("\n输入要删除的用户ID: ")
            if user_id.isdigit():
                system.delete_user(int(user_id))
            else:
                print("✗ 错误: 请输入有效的用户ID")

        elif choice == '4':  # 为用户分配角色
            user_id = input("\n输入用户ID: ")
            role_id = input("输入角色ID: ")
            if user_id.isdigit() and role_id.isdigit():
                system.assign_role_to_user(int(user_id), int(role_id))
            else:
                print("✗ 错误: 请输入有效的ID")

        elif choice == '5':  # 查看用户权限
            user_id = input("\n输入用户ID: ")
            if user_id.isdigit():
                permissions = system.get_user_permissions(int(user_id))
                display_table(["ID", "权限名", "权限代码"], permissions, ["id", "name", "code"])
            else:
                print("✗ 错误: 请输入有效的用户ID")

        elif choice == '6':  # 查看我的权限
            if system.current_user:
                permissions = system.get_user_permissions(system.current_user['id'])
                display_table(["ID", "权限名", "权限代码"], permissions, ["id", "name", "code"])

        elif choice == '7':  # 返回
            break


def role_management_menu(system: PermissionSystem):
    """角色管理菜单"""
    while True:
        choice = display_menu("角色管理", [
            "列出所有角色",
            "创建新角色",
            "为角色添加权限"
        ])

        if choice == '1':  # 列出所有角色
            roles = system.get_roles()
            display_table(["ID", "角色名"], roles, ["id", "name"])

        elif choice == '2':  # 创建新角色
            role_name = input("\n输入新角色名称: ")
            system.create_role(role_name)

        elif choice == '3':  # 为角色添加权限
            role_id = input("\n输入角色ID: ")
            permission_id = input("输入权限ID: ")
            if role_id.isdigit() and permission_id.isdigit():
                system.add_permission_to_role(int(role_id), int(permission_id))
            else:
                print("✗ 错误: 请输入有效的ID")

        elif choice == '4':  # 返回
            break


def permission_management_menu(system: PermissionSystem):
    """权限管理菜单"""
    while True:
        choice = display_menu("权限管理", [
            "列出所有权限",
            "检查用户权限"
        ])

        if choice == '1':  # 列出所有权限
            permissions = system.get_permissions()
            display_table(["ID", "权限名", "权限代码"], permissions, ["id", "name", "code"])

        elif choice == '2':  # 检查用户权限
            user_id = input("\n输入用户ID: ")
            permission_code = input("输入权限代码(如user:create): ")
            if user_id.isdigit():
                has_permission = system.check_permission(int(user_id), permission_code)
                print(f"\n用户ID {user_id} {'✓ 拥有' if has_permission else '✗ 不拥有'}权限 '{permission_code}'")
            else:
                print("✗ 错误: 请输入有效的用户ID")

        elif choice == '3':  # 返回
            break


def main_menu(system: PermissionSystem):
    """主菜单"""
    while True:
        if not system.current_user:
            if not auth_menu(system):
                return
            continue

        print(f"\n当前用户: {system.current_user['username']}")
        choice = display_menu("主菜单", [
            "用户管理",
            "角色管理",
            "权限管理",
            "登出"
        ])

        if choice == '1':  # 用户管理
            user_management_menu(system)
        elif choice == '2':  # 角色管理
            role_management_menu(system)
        elif choice == '3':  # 权限管理
            permission_management_menu(system)
        elif choice == '4':  # 登出
            system.logout()
            if not auth_menu(system):
                return
        elif choice == '5':  # 退出
            return


def main():
    """主程序入口"""
    print("=== 权限管理系统 ===")
    print("数据库配置: localhost/my_db (root/123456)")
    print("使用PyMySQL作为数据库连接器")

    system = PermissionSystem()

    try:
        main_menu(system)
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在退出...")
    finally:
        system.close()
        print("\n系统已退出")


if __name__ == "__main__":
    main()