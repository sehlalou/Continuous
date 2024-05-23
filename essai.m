rng(30); % Fix the seed for experiments
% Define the parameters
global lambda_1;
global lambda_2;
lambda_1 = 20;
lambda_2 = 20;


global t_init;
t_init = 1;
reg_model = 1; % Regularized Regression Model: LASSO (1), RIDGE (2), ELASTIC NET (3)
max_iter = 2000;


%%%%%%%%%%%%%%%%%%%%%% GENERATE DATA %%%%%%%%%%%%%%%%%%%%%

% % Load the dataset from a CSV file
% filename = 'Student_Performance.csv';  % Replace with your actual file name
% data = readtable(filename);
% 
% 
% % Initialize matrix A
% A = [];
% 
% %Convert each column to numeric if necessary and concatenate
% for i = 1:width(data)
%     column = data{:, i};  % Extract column
% 
%     if iscell(column)
%         % Convert binary categorical variables with "yes" or "no" to numeric
%         if all(ismember(column, {'yes', 'no'}))
%             column = double(strcmp(column, 'yes'));
%         else
%             % For other cell array columns, convert to categorical then to numeric
%             column = double(categorical(column));
%         end
%     end
% 
%     A = [A, column];  % Concatenate to matrix A
% end
% 
% %Extract the dependent variable (assuming it's the last column in the dataset)
% b = A(:, end);
% %b = str2double(b_cell); % Convert cell array of strings to numeric array
% A = A(:, 1:end-1);
% 
% % Display the variable types for reference
% disp('Variable types in the table:');
% disp(varfun(@class, data, 'OutputFormat', 'table'));
% 
% %disp('Matrix A:');
% %disp(A);
% 
% disp('Vector b:');
% disp(b);

% 
% % Generate synthetic data (you can replace this with your own data)
% n = 100; % Number of samples
% m = 100; % Number of features
% A = randn(n, m); % Design matrix
% true_beta = randn(m, 1); % True coefficient vector
% b = A * true_beta + 0.1 * randn(n, 1); % Target vector with noise

% Run the ISTA algorithm
% [x_old, history_ista] = ISTA(A, b, max_iter, reg_model, 1);
% [x_old, history_ista_back] = ISTA(A, b, max_iter, reg_model, 2);
% 
% % Run the FISTA algorithm
% [x_old, history_fista] = FISTA(A, b, max_iter, reg_model);
% 
% [x_old, history_fista_restart] = FISTA_restart(A, b,max_iter,reg_model);


% Run the LBFGS algorithm
%[x_old, history_lbfgs] = LBFGS(A, b, max_iter, reg_model, 1, 10);
[x_old, history_lbfgs_back] = LBFGS(A, b, max_iter, reg_model, 2, 10);

% [x_old, history_lbfgs_1] = LBFGS(A, b, max_iter, 1, 1, 10);
% [x_old, history_lbfgs_2] = LBFGS(A, b, max_iter, 1, 1, 10);
% [x_old, history_lbfgs_3] = LBFGS(A, b, max_iter, 1, 1, 10);






% Plot the error curves
figure;
%plot(0:length(history_ista.error)-1, history_ista.error, 'LineWidth', 2, 'DisplayName', 'ISTA');
hold on;
plot(0:length(history_ista_back.error)-1, history_ista_back.error, 'LineWidth', 2, 'DisplayName', 'ISTA');
plot(0:length(history_fista.error) - 1, history_fista.error, 'LineWidth', 2, 'DisplayName', 'FISTA');
%plot(0:length(history_fista_restart.error) - 1, history_fista_restart.error, 'LineWidth', 2, 'DisplayName', 'FISTA Restart');
%plot(0:length(history_lbfgs.error) - 1, history_lbfgs.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');
plot(0:length(history_lbfgs_back.error) - 1, history_lbfgs_back.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');
% plot(0:length(history_lbfgs_1.error) - 1, history_lbfgs_1.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');
% plot(0:length(history_lbfgs_2.error) - 1, history_lbfgs_2.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');
% plot(0:length(history_lbfgs_3.error) - 1, history_lbfgs_3.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');

title('Error vs. Iterations');
xlabel('Iterations');
ylabel('Error');
%legend('ISTA', 'FISTA', 'FISTA Restart');
legend('FISTA', 'FISTA restart', 'ISTA fixed step size', 'ISTA backtracking', 'L-BFGS backtracking', 'L-BFGS fixed step size');
grid on;
hold off;


function[x_old, history] = ISTA(A, b, max_iter, reg_model, step_size_method)
    
    [~, m] = size(A); % Get the number of features 
    x_old = zeros(m, 1); % Initial point x0
    history.error = [];
    global t_init;
    iter = 0;
    error = objective_function(x_old,A,b,reg_model);
    history.error = [history.error; error];
    L = max(eig(A' * A));

    for k=1:max_iter
        % Compute the gradient
        grad = A' * (A * x_old - b);

        if step_size_method == 1 % fixed size 1/L
            t = 1/L;
        elseif step_size_method == 2 % Perform Backtracking Line Search for step size
            
            t = t_init;
            c = 0.5;
            rho = 0.9;
    
           while objective_function(x_old, A, b, reg_model) - objective_function(x_old - t * grad, A, b, reg_model) < c * t * (norm(grad)^2)
                t = rho * t; 
           end
        end

       

        % Soft thresholding step
        x_new = proximal_operator(x_old - t * grad, t, reg_model);

        % Computing the error
        error = objective_function(x_old,A,b, reg_model);
        
        history.error = [history.error; error];


        iter = iter + 1;
        x_old = x_new;
    end
    
end

function [x_old, history] = FISTA(A, b, max_iter, reg_model)

    [~, m] = size(A); % Get the number of features 
    x_old = zeros(m, 1); % Initial point x0
    y_new = x_old;
    history.error = [];
    iter = 0;
    error = objective_function(x_old,A,b,reg_model);
    history.error = [history.error; error];
    t = 1;
    L = max(eig(A' * A));

    for k = 1:max_iter
        % Compute the gradient
        grad = A' * (A * y_new - b);
        
        % Update of y
        y = y_new - 1/L * grad;
        
        % Soft thresholding step

        %x_new = sign(y) .* max(abs(y) - lambda / L, 0);
        x_new = proximal_operator(y, 1/L, reg_model);
        
        % Update of t and z
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;

    
        y_new = x_new + ((t - 1) / t_new) * (x_new - x_old);
        
        % Update x_hat and t
        x_old = x_new;
        t = t_new;
        
        % Calculate error
        error = objective_function(x_old,A,b,reg_model);
        history.error = [history.error; error];
        iter = iter + 1;

    end
end 


function [x_old, history] = FISTA_restart(A, b,max_iter, reg_model)
    % Dimensions
    [m, n] = size(A);
    x_old = zeros(n, 1);
    history.error = [];
    iter = 0;
    y_new = x_old;
    error = objective_function(x_old,A,b,reg_model);
    t = 1;
    L = max(eig(A' * A));
    history.error = [history.error; error]; 
    restart_counter = 0;
    restart_iter = 2;

    for k = 1:max_iter

       
        % Gradient descent step
        grad = A' * (A * y_new - b);
        
        % Update of y
        y = y_new - 1/L * grad;
 
        % Soft thresholding step
        x_new = proximal_operator(y, 1/L, reg_model);
        

        % Update of t and y
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;
   
        y_new = x_new + ((t - 1) / t_new) * (x_new - x_old);
        
        
        % Update x_hat and t
        x_old = x_new;
        t = t_new;
        
        % Calculate error
        error = objective_function(x_old,A,b,reg_model);
        history.error = [history.error; error];
    
        if iter > 0 && history.error(end) > history.error(end - restart_iter + 1)
            % If no improvement in the last restart_iter iterations, restart
            x_old = randn(n, 1);
            %x_old = zeros(n, 1);
            t = 1;
            y_new = x_old;
            restart_counter = restart_counter + 1;
            %fprintf('Restarting FISTA at iteration %d\n', iter);
            
        end
       

        % Check convergence
        %if abs(history.error(iter+1) - history.error(iter)) < tol
        %    break;
        %end
        %fprintf('Fista v2 Iteration %d: Gradient = %f, Objective = %f\n', iter, norm(grad), error);

        iter = iter + 1;

    end
end 

function [x, history] = LBFGS(A, b, max_iter, reg_model, step_size_method, memory_lenght)
    [n, m] = size(A); % Get the number of features 
    x = zeros(m, 1); % Initial point x0
    history.error = [];
    error = objective_function(x, A, b, reg_model);
    history.error = [history.error; error];
    iter = 0;
    s_list = [];
    y_list = [];
    rho_list = [];
    %memory_lenght = 10;
    L = max(eig(A' * A));

    % Function to compute gradient
    function grad = compute_gradient(x)
        grad = A' * (A * x - b);
    end

    % Two-loop recursion to compute the search direction
    function q = two_loop_recursion(grad)
        q = grad;
        alpha = zeros(length(rho_list), 1);
        for i = length(s_list):-1:1
            alpha(i) = rho_list{i} * (s_list{i}' * q);
            q = q - alpha(i) * y_list{i};
        end
        H0k = eye(m); % Initial Hessian approximation
        r = H0k * q;
        for i = 1:length(s_list)
            beta = rho_list{i} * (y_list{i}' * r);
            r = r + s_list{i} * (alpha(i) - beta);
        end
        q = r;
    end

    % Initial gradient and error calculation
    grad = compute_gradient(x);

    while iter < max_iter
        % Compute the search direction
        if iter == 0
            d = -grad;
        else
            d = -two_loop_recursion(grad);
        end

        if step_size_method == 1 % fixed size 1/L
            alpha = 1/L;
        elseif step_size_method == 2 % Perform Backtracking Line Search for step size
            alpha = 1;
            c1 = 0.5;
            rho = 0.9;
            while objective_function(x + alpha * d, A, b, reg_model) > objective_function(x, A, b, reg_model) + c1 * alpha * (grad' * d)
                alpha = rho * alpha;
            end
        end

        % Update x
        x_new = x + alpha * d;

        % Update gradient and difference vectors
        grad_new = compute_gradient(x_new);
        s = x_new - x;
        y = grad_new - grad;

        if norm(s) > 1e-10
            s_list = [s_list, s];
            y_list = [y_list, y];
            rho_list = [rho_list, 1 / (y' * s)];
            if length(s_list) > memory_lenght
                s_list = s_list(:, 2:end);
                y_list = y_list(:, 2:end);
                rho_list = rho_list(2:end);
            end
        end

        % Update variables for next iteration
        x = x_new;
        grad = grad_new;
        error = objective_function(x, A, b, reg_model);
        history.error = [history.error; error];
        iter = iter + 1;
    end
end



function obj_val = objective_function(x, A, b, reg_model)
    global lambda_1;
    global lambda_2;
    if reg_model == 1
        % Objective function for LASSO problem
        obj_val = 0.5 * norm(A * x - b,2)^2 + lambda_1 * norm(x, 1);
    elseif reg_model == 2
        % Objective function for RIDGE problem
        obj_val = 0.5 * norm(A * x - b,2)^2 + 0.5 * lambda_2 * norm(x, 2)^2;
    elseif reg_model == 3
        % Objective function for ELASTIC NET problem
        obj_val = 0.5 * norm(A * x - b,2)^2 + lambda_1 * norm(x, 1) + 0.5 * lambda_2 * norm(x, 2)^2;
    elseif reg_model == 4
        % Function with plateau
        if norm(x, 2)^2 > 0 && norm(x, 2)^2 < 100
            obj_val = 0.5 * (norm(x, 2)^2 + 100); % Fonction quadratique avec plateau de 20 à 100
        else
            obj_val = 0.5 * (100 + 100); % Plateau
        end
    end
end


function prox = proximal_operator(x, t, reg_model)
    global lambda_1;
    global lambda_2;
    if reg_model == 1
        % LASSO
        prox = sign(x) .* max(abs(x) - lambda_1 * t, 0);
    elseif reg_model == 2
        % RIDGE
        prox = x / (1 + 2 * lambda_2 * t);
    elseif reg_model == 3
        % ELASTIC NET
        prox = proximal_operator(proximal_operator(x, t, 2), t, 1);
    elseif reg_model == 4
        % Proximal operator for the function x^2 + 100 with plateau
        if norm(x, 2)^2 > 0 && norm(x, 2)^2 < 100
            prox = x; % La fonction quadratique n'a pas de terme de régularisation
        else
            % Utilisez le proximal operator du LASSO car c'est un terme de régularisation
            prox = x;
        end
    end
end
