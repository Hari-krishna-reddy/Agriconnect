-- 8618481240---8618489990---instr(phonenumber,'124')---861848+999+0=8618489990
-- 9124664420---9999664420

-- select instr('8618481240','124');
-- select replace('8618481280','124','999')
-- 8618481240---7--861848+999+0=8618489990

-- select concat(substring('9124664420',1,instr('9124664420','124')-1),'999',substring('9124664420',instr('9124664420','124')+3));

-- Write a  MySQL query to get the details of the employees where
--  the length of the first name greater than or equal to 8.

-- select char_length('anusha') as char_length--

-- select * from employees where char_length(first_name)>=8;

-- Write a MySQL query to append '@example.com' to user_name as email field

-- select concat(cust_name,'@example.com')as email from accounts;

-- Write a  MySQL query to get the employee id, first name and hire month.
--  employee-id,f_n,m_n,l_n,hire_date--timestamp

--  select emp_id,first_name,monthname('2025-04-04 12:40:50')from employees


-- Write a  MySQL query to get the employee id,
--  email id (discard the last four characters).

--  select emp_id,substring(email,1,char_length(email)-4)

-- Write a  MySQL query to find all employees where first names are in upper case.
--   ;hari=HARI,DIVYA==DIVYA,Anusha=ANUSHA---DIVYA


--   select * from employees where first_name=upper(first_name)

--  Write a  MySQL query to extract the last 4 character of phone numbers.
--  select right(phonenumber,4)

-- .Write a  MySQL query to get the last word of the street address.--prakasam,giddalur,brahmanapalli--- 14-1=13

-- select 'prakasam,giddalur,brahmanapalli',reverse(substring(reverse('prakasam,giddalur,brahmanapalli'),1,instr(reverse('prakasam,giddalur,brahmanapalli'),',')-1));

-- Write a MySQL query to get the locations that have minimum street_address length. Of 10 characters

-- select street_address from employees where length(street_address)>=10

-- Write a MySQL query to display the first word from those job titles which contains more than one words.


-- frontend developer
-- automation tester
-- fullstack developer
-- backend developer

-- select substring(trim('     backend developer     '),1,instr(trim('     backend developer     '),' ')-1)

-- Write a  MySQL query to display the first name and salary for all employees.
--  Format the salary to be 10 characters long, left-padded with the $ symbol.
--  Label the column SALARY.


-- --  select 'harikrishna',lpad(cast( 123.678 as char),10,'$') as SALARY ;


-- --  Count the number of occurrences of the letter "e" in "Experience".


--  extract all employees whose name start with 'h'

--  left(name,1)

--  select * from employees where left(name,1)=='h';

--  'sh'----

--  select * from employees where instr(name,'sh')=0



-- Ashwin
-- Rishabh
-- Aishwarya
-- Yash
-- Akash
-- Krishna
-- Ramesh
-- Mohan
-- Rajesh
-- Vinay


-- CREATE table ex2(name varchar(30))

-- insert INTO ex2(name)values('Ashwin'),('Vinay'),('Rajesh'),('Mohan')
-- select * FROM ex2

-- select * from ex2 where instr(name,'sh')=0;

-- 9-6----0-9---valid
-- %-----so many characters
-- _--- single character

-- like '%@gmail%'                                         hari@gmail.com
--                                                        divya@yahoo.com
--                                                        anusha@gmail.com
--                                                        sudheer@meta.com


-- select * FROM customers
-- insert into customers(name,email)values('sudheer','sudheer@meta.com'),('raju','raju@yahoo.com')


select * from customers where name like '__r_';


Write a query to find all employees whose names start with "A".---'A%'
Retrieve all products from a table where the product name contains the word "fresh".---%fresh%
Find all customers whose email addresses end with "@gmail.com".---'%@gmail.com'
Write a query to fetch all students whose names have exactly five letters.----'_____'
Retrieve all orders where the order number contains the digit "7" anywhere.--'%7%'
Find all book titles that start with "The" and end with "Story".---'the%story'
Write a query to search for product names that have "e" as the second letter.---'_e%'
Retrieve all usernames that contain an underscore (_) without treating it as a wildcard.---
Find all employees whose last name starts with "S" and has at least six characters.---'S_____%'
Write a query to list all suppliers whose company name contains "food" but is not case-sensitive.---'%food%' or '%FOOD%'

CREATE TABLE customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(15)
);
INSERT INTO customers (name, email, phone) VALUES  
('Alice Johnson', 'alice123@gmail.com', '9876543210'),  
('Bob_Smith', 'bob_smith@yahoo.com', '8765432109'),  
('Charlie-Foodie', 'charlie.food@outlook.com', '7654321098'),  
('David_Foodson', 'david@foodexpress.com', '6543210987'),  
('Eva Marie', 'eva_marie@gmail.com', '5432109876');  
Write SQL queries to answer the following:

Retrieve all customers whose names start with the letter "A".
Find all customers whose email contains "food" (case-insensitive).
List all customers whose names contain an underscore (_) without treating it as a wildcard.
Retrieve customers whose phone number ends with "0987".
Find customers whose names have exactly five characters.
Retrieve all customers whose email starts with a vowel (a, e, i, o, u).
Find customers whose name contains either a dash (-) or an underscore (_).
Search for all customers whose email address has ".com" at the end.
Retrieve customers whose names have "son" anywhere in them.
Find customers whose email has the pattern gmail.com but not at the start.