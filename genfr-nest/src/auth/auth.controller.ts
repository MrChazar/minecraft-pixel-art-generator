import { Body, Controller, Get, HttpCode, HttpStatus, Post, Request, UseGuards } from "@nestjs/common";
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from "@nestjs/swagger";

import { AuthService } from "./auth.service";
import { LoginResponseDto } from "./dto/login-response.dto";
import { LoginDto } from "./dto/login.dto";
import { RegisterDto } from "./dto/register.dto";
import { AuthGuard } from "./auth.guard";

@Controller("auth")
@ApiTags("auth")
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @ApiOperation({
    summary: "Log in with an existing account",
  })
  @ApiResponse({
    status: 200,
    description: "Logged in",
  })
  @ApiResponse({
    status: 401,
    description: "Invalid credentials or account disabled",
  })
  @HttpCode(HttpStatus.OK)
  @Post("login")
  async signIn(@Body() signInDto: LoginDto): Promise<LoginResponseDto> {
    return await this.authService.signIn(signInDto.email, signInDto.password);
  }

  @ApiOperation({
    summary: "Register a new user account",
  })
  @ApiResponse({
    status: 201,
    description: "Account created",
  })
  @ApiResponse({
    status: 409,
    description: "User with the given email already exists",
  })
  @ApiResponse({
    status: 400,
    description: "Invalid request data",
  })
  @ApiResponse({
    status: 500,
    description: "Internal server error",
  })
  @HttpCode(HttpStatus.CREATED)
  @Post("register")
  async signUp(@Body() signUpDto: RegisterDto): Promise<LoginResponseDto> {
    return await this.authService.signUp({
      email: signUpDto.email,
      password: signUpDto.password,
    });
  }

  @ApiOperation({
    summary: "Get logged in user",
  })
  @ApiResponse({
    status: 200,
    description: "Get a profile of currently logged in user",
  })
  @UseGuards(AuthGuard)
  @Get('profile')
  @ApiBearerAuth("access-token")
  getProfile(@Request() request) {
    return request.user;
  }
}
