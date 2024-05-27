rng(30); % Fix the seed for experiments
% Define the parameters
global lambda_1;
global lambda_2;
lambda_1 = 0.01;
lambda_2 = 0.01;
tolerance = 0.1;
global t_init;
t_init = 1;
reg_model = 1; % Regularized Regression Model: LASSO (1), RIDGE (2), ELASTIC NET (3)


%%%%%%%%%%%%%%%%%%%%%% LOAD CSV DATA %%%%%%%%%%%%%%%%%%%%%

% Load the dataset from a CSV file

%filename = 'profit.csv';
%filename = 'Cars.csv';
%filename = 'Student_Performance.csv';
% 
% data = readtable(filename);
% 
% 
% A = [];
% for i = 1:width(data)
%     column = data{:, i};  % Extract column
% 
%     if iscell(column)
%         
%         if all(ismember(column, {'yes', 'no'}))
%             column = double(strcmp(column, 'yes'));
%         else
%            
%             column = double(categorical(column));
%         end
%     end
% 
%    A = [A, column];  % Concatenate to matrix A
% end 
% Extract the target variable (assuming it's the last column in the dataset)
% b = A(:, end);
% A = A(:, 1:end-1);

%%%%%%%%%%%%%%%%%%%%%% GENERATE SYNTHETHIC DATA %%%%%%%%%%%%%%%%%%%%%

n = 100; % Number of samples
m = 100; % Number of features
A = randn(n, m); % Design matrix
true_beta = randn(m, 1); % True coefficient vector
b = A * true_beta + 0.1 * randn(n, 1); % Target vector with noise

% Run the ISTA algorithm with fixed step size 
[x_old, history_ista] = ISTA(A, b, tolerance, reg_model, 1);
  
% Run the FISTA algorithm with fixed step size 
[x_old, history_fista] = FISTA(A, b, tolerance, reg_model);

% Run the FISTA (with restart) algorithm with fixed step size 
%[x_old, history_fista_restart] = FISTA_restart(A, b,tolerance,reg_model);

% Run the LBFGS algorithm with fixed step size
%[x_old, history_lbfgs] = LBFGS(A, b, tolerance, 2, 1, 10);

% Plot the error curves
figure;
plot(0:length(history_ista.error)-1, history_ista.error, 'LineWidth', 2, 'DisplayName', 'ISTA');
hold on;
plot(0:length(history_fista.error) - 1, history_fista.error, 'LineWidth', 2, 'DisplayName', 'FISTA');
%plot(0:length(history_fista_restart.error) - 1, history_fista_restart.error, 'LineWidth', 2, 'DisplayName', 'FISTA Restart');
%plot(0:length(history_lbfgs.error) - 1, history_lbfgs.error, 'LineWidth', 2, 'DisplayName', 'L-BFGS');


title('Error vs. Iterations for FISTA');
xlabel('Iterations');
ylabel('Error');
legend('ISTA fixed step size', 'FISTA fixed stepsize');
grid on;
hold off;


function[x_old, history] = ISTA(A, b, tolerance, reg_model, step_size_method)
    
    [~, m] = size(A); % Get the number of features 
    x_old = zeros(m, 1); % Initial point x0
     history.backtracking_iters = []; % Initialize backtracking iterations history
    history.backtracking_time = []; % Initialize backtracking time history
    history.fixed_time = []; % Initialize fixed time history
    history.main_iteration_time = []; % Initialize main iteration time history
    history.prox_time = []; 
    history.error = [];
    global t_init;
    iter = 0;
    error = objective_function(x_old,A,b,reg_model);
    history.error = [history.error; error];
    L = max(eig(A' * A));
    prev_error = error;
    % Start the timer for the entire algorithm
    tic;

    while true
        % Start the timer for main iteration
        main_iter_tic = tic;

        % Compute the gradient
        grad = A' * (A * x_old - b);

        backtracking_iters = 0; % Initialize the counter for backtracking iterations
        backtracking_tic = tic; % Start the timer for backtracking
        fixed_tic = tic; % Initialize fixed t time history
        

        if step_size_method == 1 % fixed size 1/L
            t = 1/L;
            fixed_time = toc(fixed_tic);
            history.fixed_time = [history.fixed_time; fixed_time];
        elseif step_size_method == 2 % Perform Backtracking Line Search for step size
               
            t = t_init;
            c = 0.5;
            rho = 0.9;
           
           while objective_function(x_old, A, b, reg_model) - objective_function(x_old - t * grad, A, b, reg_model) < c * t * (norm(grad)^2) % Armijo condition
                t = rho * t; 
                backtracking_iters = backtracking_iters + 1; 
           end
             % Record the time taken for backtracking
              backtracking_time = toc(backtracking_tic);
             history.backtracking_iters = [history.backtracking_iters; backtracking_iters]; % Store the count
             history.backtracking_time = [history.backtracking_time; backtracking_time]; % Store the backtracking time
        end

        prox_tic = tic;
        % Soft thresholding step
        x_new = proximal_operator(x_old - t * grad, t, reg_model);
        prox_time = toc(prox_tic);
        % Computing the error
        error = objective_function(x_old,A,b, reg_model);
        history.error = [history.error; error];
        history.prox_time = [history.prox_time; prox_time];
        % Record the time taken for the main iteration
        main_iteration_time = toc(main_iter_tic);
        history.main_iteration_time = [history.main_iteration_time; main_iteration_time]; % Store the main iteration time

        iter = iter + 1;

        % Check if the change in error is below the tolerance level
        if iter > 2 & abs(prev_error - error) < tolerance
            break;
        end
        
        prev_error = error;
        x_old = x_new;

    end

    % Stop the timer and record the elapsed time for the entire algorithm
    history.computation_time = toc;    
end

function [x_old, history] = FISTA(A, b, tolerance, reg_model)

    [~, m] = size(A); % Get the number of features 
    x_old = zeros(m, 1); % Initial point x0
    y_new = x_old;
    history.error = [];
    history.fixed_time = []; % Initialize backtracking time history
    history.main_iteration_time = []; % Initialize main iteration time history
    history.prox_time = []; 
    iter = 0;
    error = objective_function(x_old,A,b,reg_model);
    history.error = [history.error; error];
    t = 1;
    L = max(eig(A' * A));
    prev_error = error;
     % Start the timer for the entire algorithm
    tic;

    while true

        main_iter_tic = tic;
        % Compute the gradient
        grad = A' * (A * y_new - b);
        fixed_tic = tic; % Initialize fixed t time history
        t_fixed = 1/L;
        fixed_time = toc(fixed_tic);
        y = y_new - t_fixed * grad;
        % Soft thresholding step
        prox_tic = tic;
        x_new = proximal_operator(y, t_fixed, reg_model);
        prox_time = toc(prox_tic);
        % Update of t
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;
        y_new = x_new + ((t - 1) / t_new) * (x_new - x_old);
        
        % Update x_old and t
        x_old = x_new;
        t = t_new;
        
        main_iter_time = toc(main_iter_tic);

        % Calculate error
        error = objective_function(x_old,A,b,reg_model);
        history.error = [history.error; error];
        history.fixed_time = [history.fixed_time; fixed_time];
        history.prox_time = [history.prox_time; prox_time];
        history.main_iteration_time = [history.main_iteration_time;main_iter_time];

        iter = iter + 1;

        % Check if the change in error is below the tolerance level
        if abs(prev_error - error) < tolerance
            break;
        end

        prev_error = error;
        

    end
     history.computation_time = toc;    

end 


function [x_old, history] = FISTA_restart(A, b,tolerance, reg_model)
    % Dimensions
    [m, n] = size(A);
    x_old = zeros(n, 1);
    history.error = [];
    history.fixed_time = []; % Initialize backtracking time history
    history.main_iteration_time = []; % Initialize main iteration time history
    history.prox_time = []; 
    iter = 0;
    y_new = x_old;
    error = objective_function(x_old,A,b,reg_model);
    t = 1;
    L = max(eig(A' * A));
    history.error = [history.error; error]; 
    restart_counter = 0;
    restart_iter = 2;
    prev_error = error;
    tic;

    for k = 1:500
        main_iter_tic = tic; 
        % Gradient descent step
        grad = A' * (A * y_new - b);

        fixed_tic = tic; % Initialize fixed t time history
        t_fixed = 1/L;
        fixed_time = toc(fixed_tic);

        % Update of y
        y = y_new - t_fixed * grad;
        prox_tic = tic;
        % Soft thresholding step
        x_new = proximal_operator(y, t_fixed, reg_model);
        prox_time = toc(prox_tic);
        % Update of t
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;
        y_new = x_new + ((t - 1) / t_new) * (x_new - x_old);   
        % Update x_old and t
        x_old = x_new;
        t = t_new;
        
        % Calculate error
        error = objective_function(x_old,A,b,reg_model);
        history.error = [history.error; error];
           
        % If no improvement in the last restart_iter iterations, restart
        if iter > 0 && history.error(end) > history.error(end - restart_iter + 1)
            x_old = randn(n, 1);
            t = 1;
            y_new = x_old;
            restart_counter = restart_counter + 1;
           
            
        end
        
        iter = iter + 1;
        main_iter_time = toc(main_iter_tic);
        history.main_iteration_time = [history.main_iteration_time;main_iter_time];
        history.prox_time = [history.prox_time; prox_time];
        history.fixed_time = [history.fixed_time; fixed_time];
    end
    history.computation_time = toc;    
end 


function [x, history] = LBFGS(A, b, tolerance, reg_model, step_size_method, memory_lenght)
    [n, m] = size(A); % Get the number of features 
    x = zeros(m, 1); % Initial point x0
    disp(class(x));
    history.error = [];
    history.backtracking_iters = []; % Initialize backtracking iterations history
    history.backtracking_time = []; % Initialize backtracking time history
    history.fixed_time = []; % Initialize fixed time history
    history.main_iteration_time = []; % Initialize main iteration time history
    history.twoloop_time = [];
    error = objective_function(x, A, b, reg_model);
    history.error = [history.error; error];
    iter = 0;
    s_list = {};
    y_list = {};
    rho_list = {};
   
    L = max(eig(A' * A));
    prev_error = error;
    global t_init;
    % Initial gradient and error calculation
    grad = A' * (A * x - b);

    tic;
    
    while true
        % Compute the search direction
        main_iter_tic = tic;
        twoloop_tic = tic;
        d = grad;
        alpha = zeros(length(rho_list), 1);
        
        if iter == 0
            Bk = eye(m);
        else
            Bk = (s_list{end}' * y_list{end}) / (y_list{end}' * y_list{end});
        end

        for i = size(rho_list):-1:1
            disp(alpha);
            alpha(i) = rho_list{i} * (s_list{i}' * d);
            d = d - alpha(i) * y_list{i};
        end
        r = Bk * d;
        for i = 1:length(rho_list)
            disp(length(rho_list));
            disp(length(s_list));
            beta = rho_list{i} * (y_list{i}' * r);
            r = r + s_list{i} * (alpha(i) - beta);
        end
        d = -r;
        twoloop_time = toc(twoloop_tic);

        backtracking_iters = 0; % Initialize the counter for backtracking iterations
        backtracking_tic = tic; % Start the timer for backtracking

        if step_size_method == 1 % fixed size 1/L
            fixed_tic = tic; % Initialize fixed t time history
            alpha = 1/L;
            fixed_time = toc(fixed_tic);
            history.fixed_time = [history.fixed_time; fixed_time]; % Initialize fixed time history
        elseif step_size_method == 2 % Perform Backtracking Line Search for step size

            alpha = t_init;
            c = 0.5;
            rho = 0.9;
    
           while objective_function(x, A, b, reg_model) - objective_function(x - alpha * grad, A, b, reg_model) < c * alpha * (norm(grad)^2)
                alpha = rho * alpha; 
                backtracking_iters = backtracking_iters + 1; 
           end
  
            % Record the time taken for backtracking
            backtracking_time = toc(backtracking_tic);
            history.backtracking_iters = [history.backtracking_iters; backtracking_iters]; % Initialize backtracking iterations history
            history.backtracking_time = [history.backtracking_time; backtracking_time]; % Initialize backtracking time history
        end

        
        % Update x
        x_new = x + (alpha * d);

        % Update gradient and difference vectors
        grad_new = A' * (A * x_new - b);
        s = x_new - x;
        y = grad_new - grad;

        s_list = [s_list; s];
        y_list = [y_list; y];
        rho_list = [rho_list; 1 / (y' * s)];

        % V = eye(m) - (1 / (y' * s)) * y * s';
        % Bk = V' * Bk * V + (1 / (y' * s))  * s * s';


        if length(rho_list) > memory_lenght
            s_list = s_list(2:end);
            y_list = y_list(2:end);
            rho_list = rho_list(2:end);
        end

        main_iter_time = toc(main_iter_tic);
        % Update variables for next iteration
        x = x_new;
        grad = grad_new;
        error = objective_function(x, A, b, reg_model);
        history.error = [history.error; error];
        
        
        history.main_iteration_time = [history.main_iteration_time; main_iter_time]; % Initialize main iteration time history
        history.twoloop_time = [history.twoloop_time; twoloop_time];
          
         % Check if the change in error is below the tolerance level
        if abs(prev_error - error) < tolerance
            break;
        end

        prev_error = error;

        iter = iter + 1;

     
    end

     history.computation_time = toc;    
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
    end
end
