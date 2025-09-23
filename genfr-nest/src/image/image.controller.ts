import { Controller, Get, Post, Body, Patch, Param, Delete, HttpCode, HttpStatus, Req, UseGuards, Request, UnauthorizedException } from '@nestjs/common';
import { ImageService } from './image.service';
import { CreateImageDto } from './dto/create-image.dto';
import { UpdateImageDto } from './dto/update-image.dto';
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { RoleGuard } from 'src/auth/roles/role.guard';
import { Roles } from 'src/auth/roles/role.decorator';
import { Role } from '@prisma/client';
import { AuthGuard } from 'src/auth/auth.guard';
import { CreateImageResponseDto } from './dto/create-image-response.dto';

@Controller('images')
@ApiTags("images")
export class ImageController {
  constructor(private readonly imageService: ImageService) {}

  @Post()
  @HttpCode(HttpStatus.CREATED)
  @ApiOperation({
    summary: 'Generate a new image and save its data to the database',
  })
  @ApiResponse({
    status: 201,
    description: 'Image successfully generated and stored',
    type: CreateImageResponseDto,
  })
  @ApiResponse({
    status: 400,
    description: 'Invalid request body',
  })
  @UseGuards(AuthGuard)
  @ApiBearerAuth("access-token")
  async create(@Body() createImageDto: CreateImageDto, @Request() request) {
    const userEmail = request.user.email;
    return this.imageService.create(createImageDto, userEmail);
  }

  @Get()
  @ApiOperation({ summary: 'Retrieve all images' })
  @ApiResponse({
    status: 200,
    description: 'List of all stored images',
    type: [CreateImageResponseDto], 
  })
  async findAll() {
    return this.imageService.findAll();
  }

  @Get('me')
  @ApiOperation({ summary: 'Retrieve all images of current logged in user' })
  @ApiResponse({
    status: 200,
    description: 'Images data found for the given URL',
    type: [CreateImageDto],
  })
  @ApiResponse({
    status: 404,
    description: 'Images not found',
  })
  @UseGuards(AuthGuard,RoleGuard)
  @Roles(Role.ADMIN, Role.MODERATOR)
  @ApiBearerAuth("access-token")
  async findAllByUser(@Request() request) {
    return await this.imageService.findAllByUser(request.user.email);
  }

  @Get(':url')
  @ApiOperation({ summary: 'Retrieve a single image by its URL' })
  @ApiResponse({
    status: 200,
    description: 'Image data found for the given URL',
    type: CreateImageDto,
  })
  @ApiResponse({
    status: 404,
    description: 'Image not found',
  })
  async findOne(@Param('url') url: string) {
    return await this.imageService.findOne(url);
  }


  @Patch(':url')
  @ApiOperation({ summary: 'Update metadata of a specific image' })
  @ApiResponse({
    status: 200,
    description: 'Image metadata successfully updated',
    type: UpdateImageDto,
  })
  @ApiResponse({
    status: 404,
    description: 'Image not found',
  })
  @ApiBearerAuth("access-token")
  async update(@Param('url') url: string, @Body() updateImageDto: UpdateImageDto,@Request() request) {
    const previousImage = await this.imageService.findOne(url)
    if (!(request.user.role === "MODERATOR" || request.user.role === "ADMIN" || previousImage.user_email === request.user.email)){
      throw new UnauthorizedException("You are not allowed to update this image data")
    }
    return await this.imageService.update(url, updateImageDto);
  }

  @Delete(':url')
  @UseGuards(AuthGuard,RoleGuard)
  @Roles(Role.ADMIN, Role.MODERATOR)
  @ApiOperation({ summary: 'Delete an image by its URL' })
  @ApiResponse({
    status: 200,
    description: 'Image successfully deleted',
  })
  @ApiResponse({
    status: 403,
    description: 'Forbidden: only ADMIN or MODERATOR can delete images',
  })
  @ApiResponse({
    status: 404,
    description: 'Image not found',
  })
  @ApiBearerAuth("access-token")
  async remove(@Param('url') url: string, @Request() request) {
    const previousImage = await this.imageService.findOne(url)
    if (!(request.user.role === "MODERATOR" || request.user.role === "ADMIN" || previousImage.user_email === request.user.email)){
      throw new UnauthorizedException("You are not allowed to update this image data")
    }
    return await this.imageService.remove(url);
  }
}
